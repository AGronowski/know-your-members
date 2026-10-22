"""Loading members and history: the synthetic demo community, lead-provided profile files, Luma subscriber
exports (as prospects only) and Discord chat exports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import log
from .config import FIXTURES_DIR
from .db import get_setting, utcnow
from .facts import extract_answers, extract_record, replace_facts
from .graph import pair, rebuild_interactions
from .members import onboard, set_consent
from .models import CONSENT_SCOPES, Interaction, Member, Message

DEMO_DISCORD_BASE = 100_000_000_000_000_000
DEMO_CHANNEL_BASE = 400_000_000_000_000_000
DEMO_THREAD_BASE = 300_000_000_000_000_000
DEMO_MESSAGE_BASE = 200_000_000_000_000_000
# Set by `kym discord-seed` once a demo thread is mirrored into a real Discord server (value: the real thread id),
# and the webhooks it posted through (their messages are already stored under the synthetic authors).
MIRRORED_THREAD_KEY = "discord-seed:thread:{}"
SEED_WEBHOOKS_SETTING = "discord-seed:webhooks"


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("members", "profiles", "people", "data", "rows"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"{path}: no list of records found")
    return data


def seed_demo(session: Session, now: datetime | None = None, use_llm: bool = False) -> dict:
    """Load the synthetic demo community (fixtures/), its chat history and who-knows-whom."""
    now = now or utcnow()
    spec = json.loads((FIXTURES_DIR / "demo_members.json").read_text())
    by_name: dict[str, Member] = {}
    for i, row in enumerate(spec["members"]):
        discord_id = DEMO_DISCORD_BASE + i + 1
        member = session.scalar(select(Member).where(Member.discord_id == discord_id))
        if member is None:
            member = Member(discord_id=discord_id, name=row["name"], source="synthetic", status="active",
                            handles={}, consent={}, meta={}, joined_at=now - timedelta(days=60 - i))
            session.add(member)
            session.flush()
        member.responsiveness = row.get("responsiveness", 0.7)
        member.intro_frequency = row.get("frequency", "weekly")
        scopes = set(row.get("consent") or CONSENT_SCOPES)
        if {"store", "intro"} <= scopes:
            onboard(session, member, row["answers"], scopes, source="seed:demo", use_llm=use_llm)
        else:
            set_consent(session, member, {s: s in scopes for s in CONSENT_SCOPES}, "seed:demo")
            extraction = extract_answers(session, row["answers"], use_llm=use_llm)
            replace_facts(session, member, extraction.facts, {"answer"})
            member.meta = {"would_speak": extraction.would_speak}
        by_name[row["name"]] = member
    messages = load_demo_messages(session, by_name, now)
    rebuild_interactions(session)
    for a_name, b_name in spec.get("already_know_each_other", []):
        a, b = pair(by_name[a_name].id, by_name[b_name].id)
        if session.scalar(select(Interaction).where(Interaction.member_a_id == a, Interaction.member_b_id == b,
                                                    Interaction.source == "seed")) is None:
            session.add(Interaction(member_a_id=a, member_b_id=b, weight=5.0, last_at=now, source="seed"))
    log.record(session, "seed", "ok", f"{len(by_name)} synthetic members, {messages} messages", target="demo")
    return {"members": len(by_name), "messages": messages}


def load_demo_messages(session: Session, by_name: dict[str, Member], now: datetime) -> int:
    spec = json.loads((FIXTURES_DIR / "demo_messages.json").read_text())
    channel_ids = {name: DEMO_CHANNEL_BASE + i + 1 for i, name in enumerate(spec["channels"])}
    count = 0
    for t, thread in enumerate(spec["threads"]):
        thread_id = DEMO_THREAD_BASE + t + 1
        if get_setting(session, MIRRORED_THREAD_KEY.format(thread_id)):
            continue  # already lives in Discord under real ids
        ids = [DEMO_MESSAGE_BASE + thread_id % 1_000_000 * 100 + i for i in range(len(thread["messages"]))]
        for i, msg in enumerate(thread["messages"]):
            if session.get(Message, ids[i]):
                continue
            author = by_name[msg["author"]]
            session.add(Message(
                id=ids[i], guild_id=None, channel_id=channel_ids[thread["channel"]], channel_name=thread["channel"],
                thread_id=thread_id, thread_name=thread["name"], author_id=author.discord_id, author_name=author.name,
                author_is_bot=False, content=msg["content"], created_at=now - timedelta(hours=msg["hours_ago"]),
                reply_to_id=ids[msg["reply_to"]] if "reply_to" in msg else None, mention_ids=[], jump_url=None))
            count += 1
    session.flush()
    return count


def import_members(session: Session, path: Path, *, assume_consent: bool, use_llm: bool = True, limit: int | None = None) -> dict:
    """Import a profile file (CSV/JSON/JSONL, any columns). Without --assume-consent rows become prospects:
    stored, but never matched, enriched or shown to anyone until they opt in on Discord."""
    rows = read_rows(path)[:limit] if limit else read_rows(path)
    created = updated = skipped = 0
    for row in rows:
        extraction = extract_record(session, row, source="import", use_llm=use_llm)
        name = extraction.name
        if not name:
            skipped += 1
            continue
        email = next((str(v).strip().lower() for k, v in row.items() if "email" in str(k).lower() and v), None)
        member = session.scalar(select(Member).where(Member.email == email)) if email else None
        member = member or session.scalar(select(Member).where(Member.name == name, Member.source == "import"))
        if member is None:
            member = Member(name=name, email=email, source="import", status="active" if assume_consent else "prospect",
                            handles={}, consent={}, meta={})
            session.add(member)
            session.flush()
            created += 1
        else:
            updated += 1
        member.handles = {**(member.handles or {}), **extraction.handles}
        member.meta = {**(member.meta or {}), "would_speak": extraction.would_speak, "extractor": extraction.extractor}
        replace_facts(session, member, extraction.facts, {"import"})
        if assume_consent:
            set_consent(session, member, {s: True for s in CONSENT_SCOPES}, f"import:{path.name}")
    log.record(session, "import", "ok", f"{created} created, {updated} updated, {skipped} skipped (no name)",
               target=path.name, assume_consent=assume_consent)
    return {"created": created, "updated": updated, "skipped": skipped}


def import_prospects(session: Session, path: Path) -> dict:
    """A Luma subscriber export (names + emails). Stored as prospects for coverage stats; no profiling."""
    created = 0
    for row in read_rows(path):
        email = next((str(v).strip().lower() for k, v in row.items() if "email" in str(k).lower() and v), None)
        name = next((str(row[k]).strip() for k in row if str(k).lower() in ("name", "full name", "full_name") and row[k]), None)
        if not email or session.scalar(select(Member).where(Member.email == email)):
            continue
        session.add(Member(name=name or email.split("@")[0], email=email, source="luma_export", status="prospect",
                           handles={}, consent={}, meta={}))
        created += 1
    log.record(session, "import_prospects", "ok", f"{created} prospects", target=path.name)
    return {"created": created}


def import_discord_export(session: Session, path: Path) -> int:
    """DiscordChatExporter JSON (one channel or thread per file) into the message mirror."""
    data = json.loads(path.read_text())
    channel = data.get("channel", {})
    channel_id = int(channel.get("id", 0))
    is_thread = channel.get("type", "").lower().endswith("thread")
    parent_id = int(channel.get("categoryId") or channel_id) if is_thread else channel_id
    members = {m.discord_id: m for m in session.scalars(select(Member).where(Member.discord_id.is_not(None)))}
    count = 0
    for msg in data.get("messages", []):
        mid = int(msg["id"])
        if session.get(Message, mid):
            continue
        author = msg.get("author", {})
        author_id = int(author.get("id", 0))
        created = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
        ref = (msg.get("reference") or {}).get("messageId")
        session.add(Message(
            id=mid, guild_id=int(data.get("guild", {}).get("id", 0)) or None, channel_id=parent_id,
            channel_name=channel.get("category") if is_thread else channel.get("name"),
            thread_id=channel_id if is_thread else None, thread_name=channel.get("name") if is_thread else None,
            author_id=author_id, author_name=(members.get(author_id).name if author_id in members else
                                              author.get("nickname") or author.get("name") or "unknown"),
            author_is_bot=bool(author.get("isBot")), content=msg.get("content") or "", created_at=created,
            reply_to_id=int(ref) if ref else None, mention_ids=[int(m["id"]) for m in msg.get("mentions", [])],
            jump_url=None))
        count += 1
    rebuild_interactions(session)
    log.record(session, "import_discord_export", "ok", f"{count} messages", target=path.name)
    return count
