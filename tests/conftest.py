import os
from datetime import datetime, timezone

import pytest

# Deterministic, offline: hashing embeddings, no LLM keys, a throwaway database per test.
os.environ["KYM_EMBEDDINGS"] = "hash"
for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "XAI_API_KEY", "NOTION_TOKEN", "DISCORD_TOKEN", "KYM_CLAUDE"):
    os.environ.pop(key, None)

from kym import config, db  # noqa: E402
from kym.config import get_settings  # noqa: E402

# Never read the developer's .env: a real bot token, Grok key or demo flags there would leak into the tests.
config.load_dotenv = lambda *args, **kwargs: False

MONDAY = datetime(2026, 10, 26, 10, 0, tzinfo=timezone.utc)  # after Monday 09:00 London (week 2026-W44)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("KYM_HTTP_CACHE", str(tmp_path / "http_cache.db"))
    get_settings.cache_clear()
    from kym.enrich import http

    monkeypatch.setattr(http, "_cache_factory", None)
    monkeypatch.setattr(http, "_fetcher", None)
    db.configure()
    db.init_db()
    yield
    get_settings.cache_clear()


@pytest.fixture
def session():
    with db.session_scope() as s:
        yield s


@pytest.fixture
def demo(session):
    from kym.importer import seed_demo

    seed_demo(session, now=MONDAY, use_llm=False)
    session.flush()
    return session
