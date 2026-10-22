"""Polite, cached HTTP for public data sources.

Per-host spacing and backoff (Luma rate-limits after a handful of quick requests), a persistent response cache
(so a demo never depends on a live call), and stale-cache fallback when a source is down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlparse

import httpx
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from .. import db
from ..config import DATA_DIR, get_settings
from ..models import HttpCache

logger = logging.getLogger("kym.http")


@dataclass(frozen=True)
class HostRule:
    min_interval: float
    ttl_hours: float


RULES = {
    "api.lu.ma": HostRule(3.0, 24 * 7),
    "api.github.com": HostRule(0.8, 24 * 3),
    "devpost.com": HostRule(1.5, 24 * 7),
    "hn.algolia.com": HostRule(0.3, 24 * 3),
}
DEFAULT_RULE = HostRule(1.0, 24)


class FetchError(RuntimeError):
    pass


_cache_factory: sessionmaker[Session] | None = None


def _cache_sessions() -> sessionmaker[Session]:
    """The HTTP cache lives in its own SQLite file, so a caller's open write transaction on the main database
    (SQLite allows one writer) can never block a fetch."""
    global _cache_factory
    if _cache_factory is None:
        path = os.getenv("KYM_HTTP_CACHE") or str(DATA_DIR / "http_cache.db")
        engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 10})

        @event.listens_for(engine, "connect")
        def _wal(dbapi_conn, _record):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

        HttpCache.__table__.create(engine, checkfirst=True)
        _cache_factory = sessionmaker(engine, expire_on_commit=False)
    return _cache_factory


def _looks_rate_limited(host: str, r: httpx.Response) -> bool:
    if r.status_code == 429:
        return True
    if host == "api.github.com" and r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0":
        return True
    return host == "api.lu.ma" and r.status_code >= 400 and "rate limit" in r.text[:300].lower()


class Fetcher:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=25.0, follow_redirects=True,
                                              headers={"User-Agent": "kym-community-bot/0.1 (community speaker finder)"})
        self._last: dict[str, float] = {}
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self.stats = defaultdict(int)

    def _headers(self, host: str, extra: dict | None) -> dict:
        headers = dict(extra or {})
        token = get_settings().github_token
        if host == "api.github.com":
            headers.setdefault("Accept", "application/vnd.github+json")
            headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def _cache_read(self, key: str) -> HttpCache | None:
        try:
            with _cache_sessions()() as session:
                row = session.get(HttpCache, key)
                if row:
                    session.expunge(row)
                return row
        except OperationalError:
            return None

    def _cache_write(self, key: str, url: str, status: int, body: str) -> None:
        try:
            with _cache_sessions()() as session, session.begin():
                row = session.get(HttpCache, key)
                if row:
                    row.status, row.body, row.fetched_at = status, body, db.utcnow()
                else:
                    session.add(HttpCache(key=key, url=url, status=status, body=body, fetched_at=db.utcnow()))
        except OperationalError as exc:  # cache is best-effort; never fail a fetch over it
            logger.warning("cache write skipped for %s: %s", url[:120], str(exc)[:120])

    def get(self, url: str, *, headers: dict | None = None, as_json: bool = True, max_age_hours: float | None = None):
        """Returns parsed JSON (or text); None for 404. Raises FetchError when the source is unavailable and
        nothing is cached."""
        host = urlparse(url).hostname or ""
        rule = RULES.get(host, DEFAULT_RULE)
        key = hashlib.sha256(url.encode()).hexdigest()
        cached = self._cache_read(key)
        max_age = timedelta(hours=max_age_hours if max_age_hours is not None else rule.ttl_hours)
        if cached and db.utcnow() - cached.fetched_at < max_age:
            self.stats["cache_hits"] += 1
            return self._decode(cached.status, cached.body, as_json)
        response, error = None, None
        with self._locks[host]:
            for attempt in range(4):
                wait = rule.min_interval - (time.monotonic() - self._last.get(host, 0.0))
                if wait > 0:
                    time.sleep(wait)
                try:
                    attempt_response = self._client.get(url, headers=self._headers(host, headers))
                except httpx.HTTPError as exc:
                    error, attempt_response = f"{type(exc).__name__}: {exc}", None
                self._last[host] = time.monotonic()
                self.stats["requests"] += 1
                if attempt_response is not None and not _looks_rate_limited(host, attempt_response) \
                        and attempt_response.status_code < 500:
                    response = attempt_response
                    break
                retry_after = None
                if attempt_response is not None:
                    limited = _looks_rate_limited(host, attempt_response)
                    error = f"HTTP {attempt_response.status_code}" + (" (rate limited)" if limited else "")
                    reset = attempt_response.headers.get("x-ratelimit-reset")
                    if host == "api.github.com" and reset and float(reset) - time.time() > 90:
                        error += "; GitHub quota exhausted (set GITHUB_TOKEN)"
                        break  # don't hang for up to an hour
                    retry_after = attempt_response.headers.get("retry-after")
                if attempt < 3:
                    time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 5.0 * (2 ** attempt))
        if response is None:
            self.stats["failures"] += 1
            if cached:
                self.stats["stale_hits"] += 1
                logger.warning("%s unavailable (%s); using cache from %s", host, error, cached.fetched_at)
                return self._decode(cached.status, cached.body, as_json)
            raise FetchError(f"{host}: {error}")
        if response.status_code == 404:
            self._cache_write(key, url, 404, "")
            return None
        if response.status_code >= 400:
            raise FetchError(f"{host}: HTTP {response.status_code} {response.text[:200]}")
        self._cache_write(key, url, response.status_code, response.text)
        return self._decode(response.status_code, response.text, as_json)

    @staticmethod
    def _decode(status: int, body: str, as_json: bool):
        if status == 404:
            return None
        return json.loads(body) if as_json else body


_fetcher: Fetcher | None = None


def fetcher() -> Fetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = Fetcher()
    return _fetcher
