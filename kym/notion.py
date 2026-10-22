"""Notion as the lead's view: Members, Intros (with a Veto checkbox the bot reads back), Revivals and the Bot log,
plus a page per speaker shortlist and per weekly digest.

Uses the REST API pinned to Notion-Version 2022-06-28 (classic database endpoints; Notion keeps old versions
working). Share one parent page with the integration and put its id in NOTION_PARENT_PAGE_ID.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import cycle, log
from .config import get_settings
from .db import get_setting, set_setting, utcnow
from .models import Attempt, Fact, Intro, Member, Revival

API = "https://api.notion.com/v1"

DATABASES = {
    "members": ("KYM Members", {
        "Name": {"title": {}}, "Status": {"select": {}}, "Building": {"rich_text": {}}, "Can help with": {"rich_text": {}},
        "Wants": {"rich_text": {}}, "Would speak": {"checkbox": {}}, "Intro frequency": {"select": {}},
        "Joined": {"date": {}}, "KYM ID": {"number": {}}}),
    "intros": ("KYM Intros", {
        "Pair": {"title": {}}, "Week": {"select": {}}, "Status": {"select": {}}, "Confidence": {"select": {}},
        "Score": {"number": {}}, "Why": {"rich_text": {}}, "Veto": {"checkbox": {}}, "Outcome": {"select": {}},
        "Writer": {"select": {}}, "KYM ID": {"number": {}}}),
    "revivals": ("KYM Thread revivals", {
        "Thread": {"title": {}}, "Rule": {"select": {}}, "Status": {"select": {}}, "Judge score": {"number": {}},
        "Draft": {"rich_text": {}}, "Why": {"rich_text": {}}, "Replies (7d)": {"number": {}}, "KYM ID": {"number": {}}}),
    "log": ("KYM Bot log", {
        "Event": {"title": {}}, "Kind": {"select": {}}, "Status": {"select": {}}, "At": {"date": {}},
        "Detail": {"rich_text": {}}, "KYM ID": {"number": {}}}),
}


class NotionError(RuntimeError):
    pass


def configured() -> bool:
    s = get_settings()
    return bool(s.notion_token and s.notion_parent_page_id)


def _text(value: str | None, limit: int = 1900) -> list[dict]:
    value = (value or "")[:limit]
    return [{"type": "text", "text": {"content": value}}] if value else []


def _select(value: str | None) -> dict:
    return {"select": {"name": value.replace(",", " ")[:100]}} if value else {"select": None}


class Notion:
    def __init__(self) -> None:
        s = get_settings()
        if not configured():
            raise NotionError("NOTION_TOKEN and NOTION_PARENT_PAGE_ID are not set")
        self.parent = s.notion_parent_page_id
        self.client = httpx.Client(base_url=API, timeout=30.0, headers={
            "Authorization": f"Bearer {s.notion_token}", "Notion-Version": s.notion_version, "Content-Type": "application/json"})
        self._last = 0.0

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        for attempt in range(5):
            wait = 0.35 - (time.monotonic() - self._last)  # Notion allows ~3 requests/second
            if wait > 0:
                time.sleep(wait)
            r = self.client.request(method, path, json=body)
            self._last = time.monotonic()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("retry-after") or 2 ** attempt))
                continue
            if r.status_code >= 400:
                raise NotionError(f"{method} {path}: HTTP {r.status_code} {r.text[:300]}")
            return r.json()
        raise NotionError(f"{method} {path}: retries exhausted")

    def create_database(self, title: str, properties: dict) -> str:
        return self.request("POST", "/databases", {"parent": {"type": "page_id", "page_id": self.parent},
                                                   "title": _text(title), "properties": properties})["id"]

    def upsert_row(self, database_id: str, page_id: str | None, properties: dict) -> str:
        if page_id:
            self.request("PATCH", f"/pages/{page_id}", {"properties": properties})
            return page_id
        return self.request("POST", "/pages", {"parent": {"database_id": database_id}, "properties": properties})["id"]

    def child_page(self, title: str, blocks: list[dict]) -> str:
        page = self.request("POST", "/pages", {"parent": {"page_id": self.parent},
                                               "properties": {"title": {"title": _text(title)}}, "children": blocks[:100]})
        for start in range(100, len(blocks), 100):
            self.request("PATCH", f"/blocks/{page['id']}/children", {"children": blocks[start:start + 100]})
        return page["url"]


def _db_id(session: Session, key: str) -> str | None:
    return get_setting(session, f"notion:db:{key}")


def setup(session: Session) -> dict:
    notion = Notion()
    created = {}
    for key, (title, props) in DATABASES.items():
        if not _db_id(session, key):
            set_setting(session, f"notion:db:{key}", notion.create_database(title, props))
            created[key] = title
    log.record(session, "notion_setup", "ok", f"created {list(created.values()) or 'nothing (already set up)'}")
    return created


def sync(session: Session, since_days: int = 56) -> dict:
    """Push members, intros, revivals and new log rows. Safe to run repeatedly (upserts by stored page ids)."""
    notion = Notion()
    if not all(_db_id(session, k) for k in DATABASES):
        setup(session)
    counts = {"members": 0, "intros": 0, "revivals": 0, "log": 0}
    db_members = _db_id(session, "members")
    for m in session.scalars(select(Member).where(Member.status == "active")):
        if not (m.consent or {}).get("store"):
            continue  # never show people who haven't opted in
        facts = session.scalars(select(Fact).where(Fact.member_id == m.id)).all()
        props = {
            "Name": {"title": _text(m.name)}, "Status": _select(m.status), "Intro frequency": _select(m.intro_frequency),
            "Building": {"rich_text": _text("; ".join(f.text for f in facts if f.kind == "building"))},
            "Can help with": {"rich_text": _text("; ".join(f.text for f in facts if f.kind == "offers"))},
            "Wants": {"rich_text": _text("; ".join(f.text for f in facts if f.kind == "wants"))},
            "Would speak": {"checkbox": bool((m.meta or {}).get("would_speak")) and m.allows("speaker")},
            "Joined": {"date": {"start": (m.onboarded_at or m.joined_at).date().isoformat()}},
            "KYM ID": {"number": m.id},
        }
        m.notion_page_id = notion.upsert_row(db_members, m.notion_page_id, props)
        counts["members"] += 1
    db_intros = _db_id(session, "intros")
    cutoff = utcnow() - timedelta(days=since_days)
    for i in session.scalars(select(Intro).where(Intro.created_at >= cutoff)):
        why = " | ".join(c["text"] for c in (i.copy or {}).get("to_a", []))
        props = {
            "Pair": {"title": _text(f"{i.member_a.name} × {i.member_b.name}")}, "Week": _select(i.week),
            "Status": _select(i.status), "Confidence": _select(i.confidence), "Score": {"number": round(i.score, 3)},
            "Why": {"rich_text": _text(why)}, "Outcome": _select(i.outcome), "Writer": _select((i.copy or {}).get("writer")),
            "KYM ID": {"number": i.id},
        }
        if not i.notion_page_id:
            props["Veto"] = {"checkbox": False}
        i.notion_page_id = notion.upsert_row(db_intros, i.notion_page_id, props)
        counts["intros"] += 1
    db_rev = _db_id(session, "revivals")
    for r in session.scalars(select(Revival).where(Revival.detected_at >= cutoff)):
        props = {
            "Thread": {"title": _text(r.title[:200])}, "Rule": _select(r.rule), "Status": _select(r.status),
            "Judge score": {"number": (r.judge or {}).get("score")}, "Draft": {"rich_text": _text(r.draft)},
            "Why": {"rich_text": _text(r.failure_reason or (r.judge or {}).get("reason"))},
            "Replies (7d)": {"number": r.replies_7d}, "KYM ID": {"number": r.id},
        }
        r.notion_page_id = notion.upsert_row(db_rev, r.notion_page_id, props)
        counts["revivals"] += 1
    db_log = _db_id(session, "log")
    for a in session.scalars(select(Attempt).where(Attempt.notion_page_id.is_(None), Attempt.at >= cutoff).order_by(Attempt.at).limit(200)):
        props = {"Event": {"title": _text(f"{a.kind}: {a.target}"[:200])}, "Kind": _select(a.kind), "Status": _select(a.status),
                 "At": {"date": {"start": a.at.isoformat()}}, "Detail": {"rich_text": _text(a.detail)}, "KYM ID": {"number": a.id}}
        a.notion_page_id = notion.upsert_row(db_log, None, props)
        counts["log"] += 1
    return counts


def apply_vetoes(session: Session, now: datetime) -> int:
    """Read the Veto checkbox on proposed intros before they are sent."""
    if not configured():
        return 0
    notion = Notion()
    vetoed = 0
    for intro in session.scalars(select(Intro).where(Intro.status == "proposed", Intro.notion_page_id.is_not(None))):
        page = notion.request("GET", f"/pages/{intro.notion_page_id}")
        if page.get("properties", {}).get("Veto", {}).get("checkbox"):
            vetoed += cycle.veto(session, intro, "lead (Notion)", now)
    return vetoed


def _bullet(text: str, url: str | None = None) -> dict:
    rich = [{"type": "text", "text": {"content": text[:1900], **({"link": {"url": url}} if url else {})}}]
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rich}}


def _para(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _text(text)}}


def _heading(text: str, level: int = 2) -> dict:
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": _text(text)}}


def publish_speakers(session: Session, result: dict) -> str:
    blocks = [_para(f"Keywords: {', '.join(result['expansion']['keywords'])}. People considered: {result.get('people_considered', 0)}.")]
    for note in result.get("notes", []):
        blocks.append(_bullet(f"Note: {note}"))
    for c in result["shortlist"]:
        blocks.append(_heading(f"{c['name']}{' (member)' if c['member'] else ''}: score {c['score']}", 3))
        blocks.append(_para(c.get("why", "")))
        for e in c["evidence"]:
            blocks.append(_bullet(f"{e['source']}: {e['text']}" + (f" ({e['date']})" if e.get("date") else ""), e.get("url")))
        blocks.append(_bullet(f"London: {c['london']['basis']}"))
        for line in c.get("identity", []):
            blocks.append(_bullet(f"Identity: {line}"))
        for line in c.get("unverified", []):
            blocks.append(_bullet(f"Unverified: {line}"))
        for label, url in c.get("profiles", {}).items():
            blocks.append(_bullet(label, url))
    url = Notion().child_page(f"Speakers: {result['topic']} ({utcnow():%d %b %Y})", blocks)
    log.record(session, "notion_speakers", "ok", url, target=result["topic"])
    return url


def publish_digest(session: Session, digest: dict, markdown_lines: list[str]) -> str:
    blocks = [_bullet(line.lstrip("•- ").strip()) if line.startswith(("•", "-")) else _para(line) for line in markdown_lines if line.strip()]
    url = Notion().child_page(f"Weekly digest {digest['week']}", blocks)
    log.record(session, "notion_digest", "ok", url, target=digest["week"])
    return url
