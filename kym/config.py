"""Runtime settings, read once from the environment (and .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIXTURES_DIR = ROOT / "fixtures"


def _int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str
    timezone: str

    discord_token: str | None
    discord_guild_id: int | None
    intros_channel_id: int | None
    lead_channel_id: int | None

    xai_api_key: str | None
    xai_model: str
    xai_base_url: str

    claude_enabled: bool
    claude_model: str
    claude_extract_effort: str
    claude_judge_effort: str

    embed_model: str
    llm_rerank: bool
    include_synthetic: bool

    github_token: str | None
    notion_token: str | None
    notion_parent_page_id: str | None
    notion_version: str

    # Weekly schedule, in local time (Monday=0).
    propose_weekday: int
    propose_hour: int
    send_weekday: int
    send_hour: int
    digest_weekday: int
    digest_hour: int
    response_ttl_hours: int
    followup_after_hours: int
    outcome_after_hours: int

    # Thread reviver
    revive_dead_after_hours: int
    revive_max_posts_per_day: int
    revive_autopost: bool
    history_days: int

    # False: nothing runs on the schedule; the lead drives each step with /kym run (live demos, rehearsals).
    autopilot: bool

    @property
    def grok_enabled(self) -> bool:
        return bool(self.xai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    DATA_DIR.mkdir(exist_ok=True)
    anthropic_creds = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    return Settings(
        database_url=os.getenv("DATABASE_URL") or f"sqlite:///{DATA_DIR / 'kym.db'}",
        timezone=os.getenv("KYM_TIMEZONE", "Europe/London"),
        discord_token=os.getenv("DISCORD_TOKEN") or None,
        discord_guild_id=_int("DISCORD_GUILD_ID"),
        intros_channel_id=_int("KYM_INTROS_CHANNEL_ID"),
        lead_channel_id=_int("KYM_LEAD_CHANNEL_ID"),
        xai_api_key=os.getenv("XAI_API_KEY") or None,
        xai_model=os.getenv("XAI_MODEL", "grok-4.7"),
        xai_base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        # The SDK can also authenticate from an `ant auth login` profile; KYM_CLAUDE=1 opts in then.
        claude_enabled=_bool("KYM_CLAUDE", anthropic_creds),
        claude_model=os.getenv("KYM_CLAUDE_MODEL", "claude-opus-5-5"),
        claude_extract_effort=os.getenv("KYM_CLAUDE_EXTRACT_EFFORT", "low"),
        claude_judge_effort=os.getenv("KYM_CLAUDE_JUDGE_EFFORT", "medium"),
        embed_model=os.getenv("KYM_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        llm_rerank=_bool("KYM_LLM_RERANK", True),
        include_synthetic=_bool("KYM_INCLUDE_SYNTHETIC", False),
        github_token=os.getenv("GITHUB_TOKEN") or None,
        notion_token=os.getenv("NOTION_TOKEN") or None,
        notion_parent_page_id=os.getenv("NOTION_PARENT_PAGE_ID") or None,
        notion_version=os.getenv("NOTION_VERSION", "2022-06-28"),
        propose_weekday=_int("KYM_PROPOSE_WEEKDAY", 0),
        propose_hour=_int("KYM_PROPOSE_HOUR", 9),
        send_weekday=_int("KYM_SEND_WEEKDAY", 1),
        send_hour=_int("KYM_SEND_HOUR", 9),
        digest_weekday=_int("KYM_DIGEST_WEEKDAY", 4),
        digest_hour=_int("KYM_DIGEST_HOUR", 17),
        response_ttl_hours=_int("KYM_RESPONSE_TTL_HOURS", 72),
        followup_after_hours=_int("KYM_FOLLOWUP_AFTER_HOURS", 96),
        outcome_after_hours=_int("KYM_OUTCOME_AFTER_HOURS", 168),
        revive_dead_after_hours=_int("KYM_REVIVE_DEAD_AFTER_HOURS", 72),
        revive_max_posts_per_day=_int("KYM_REVIVE_MAX_POSTS_PER_DAY", 3),
        revive_autopost=_bool("KYM_REVIVE_AUTOPOST", False),
        history_days=_int("KYM_HISTORY_DAYS", 60),
        autopilot=_bool("KYM_AUTOPILOT", True),
    )
