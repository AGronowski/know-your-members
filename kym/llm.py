"""LLM calls with JSON-schema outputs.

Roles: Grok writes (intros, nudges, speaker blurbs); Claude extracts and judges. Each role falls back to the other
provider when its first choice isn't configured or fails, and callers fall back to deterministic code when neither
is available (LLMUnavailable) or both fail (LLMFailed). Nothing here decides on its own: callers verify outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import lru_cache

import httpx
from sqlalchemy.orm import Session

from .config import get_settings
from .models import LLMCache

logger = logging.getLogger("kym.llm")

ROLE_ORDER = {"write": ("grok", "claude"), "extract": ("claude", "grok"), "judge": ("claude", "grok")}


class LLMUnavailable(RuntimeError):
    """No provider is configured for this role."""


class LLMFailed(RuntimeError):
    """Providers were configured but every call failed, was refused, or returned invalid JSON."""


def available(provider: str) -> bool:
    s = get_settings()
    return s.claude_enabled if provider == "claude" else s.grok_enabled


def any_available() -> bool:
    return available("claude") or available("grok")


def providers_for(role: str) -> list[str]:
    return [p for p in ROLE_ORDER[role] if available(p)]


def cache_key(role: str, system: str, user: str, schema: dict) -> str:
    blob = json.dumps([role, system, user, schema], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_get(session: Session, key: str) -> dict | None:
    row = session.get(LLMCache, key)
    return dict(row.response) if row else None


def cache_put(session: Session, key: str, provider: str, data: dict) -> None:
    if session.get(LLMCache, key) is None:
        session.add(LLMCache(key=key, provider=provider, response=data))


@lru_cache(maxsize=1)
def _anthropic():
    import anthropic

    return anthropic.Anthropic(max_retries=3, timeout=180.0)


def _claude(system: str, user: str, schema: dict, effort: str) -> dict:
    import anthropic

    s = get_settings()
    try:
        response = _anthropic().beta.messages.create(
            model=s.claude_model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.RateLimitError as exc:
        raise LLMFailed(f"claude rate limited: {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMFailed(f"claude HTTP {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMFailed(f"claude connection error: {exc}") from exc
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise LLMFailed(f"claude refused (category={category})")
    if response.stop_reason == "max_tokens":
        raise LLMFailed("claude hit max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise LLMFailed("claude returned no text block")
    return json.loads(text)


def _grok(system: str, user: str, schema: dict, name: str) -> dict:
    s = get_settings()
    body = {
        "model": s.xai_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
    }
    delay = 2.0
    for attempt in range(4):
        try:
            r = httpx.post(f"{s.xai_base_url}/chat/completions", json=body, timeout=180.0,
                           headers={"Authorization": f"Bearer {s.xai_api_key}"})
        except httpx.HTTPError as exc:
            if attempt == 3:
                raise LLMFailed(f"grok connection error: {exc}") from exc
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == 3:
                raise LLMFailed(f"grok HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(float(r.headers.get("retry-after") or delay))
            delay *= 2
            continue
        if r.status_code >= 400:
            raise LLMFailed(f"grok HTTP {r.status_code}: {r.text[:300]}")
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMFailed("grok hit its length limit")
        content = choice["message"].get("content")
        if not content:
            raise LLMFailed("grok returned no content")
        return json.loads(content)
    raise LLMFailed("grok: retries exhausted")


def call(role: str, system: str, user: str, schema: dict, *, name: str = "result") -> tuple[dict, str]:
    """One structured call, trying the role's providers in order. Thread-safe; no DB access."""
    providers = providers_for(role)
    if not providers:
        raise LLMUnavailable(f"no LLM configured for {role}")
    s = get_settings()
    errors = []
    for provider in providers:
        try:
            if provider == "claude":
                effort = s.claude_extract_effort if role == "extract" else s.claude_judge_effort
                return _claude(system, user, schema, effort), provider
            return _grok(system, user, schema, name), provider
        except (LLMFailed, json.JSONDecodeError, KeyError) as exc:
            logger.warning("%s via %s failed: %s", role, provider, exc)
            errors.append(f"{provider}: {exc}")
    raise LLMFailed("; ".join(errors))


def complete_json(session: Session | None, role: str, system: str, user: str, schema: dict,
                  *, name: str = "result") -> tuple[dict, str]:
    """`call` plus the persistent cache (same prompt -> same answer, no second bill)."""
    key = cache_key(role, system, user, schema)
    if session is not None:
        cached = cache_get(session, key)
        if cached is not None:
            return cached["data"], cached["provider"] + ":cached"
    data, provider = call(role, system, user, schema, name=name)
    if session is not None:
        cache_put(session, key, provider, {"data": data, "provider": provider})
    return data, provider
