"""Synchronous database operations the bot runs via asyncio.to_thread. They return plain dicts, never ORM objects."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .. import cycle, db, digest, log, notion, reviver
from ..config import get_settings
from ..db import get_setting, session_scope, set_setting
from ..enrich import enrich_member
from ..graph import rebuild_interactions
from ..importer import SEED_WEBHOOKS_SETTING
from ..members import QUESTIONS, forget, get_or_create_discord_member, onboard, profile_summary, set_frequency
from ..models import Answer, Intro, Member, Message, Revival
from ..speakers import find_speakers
from ..writer import lead_lines, render_intro


def channel_setting(name: str) -> int | None:
    with session_scope() as session:
        value = get_setting(session, f"discord:{name}_channel")
    if value:
        return int(value)
    s = get_settings()
    return s.intros_channel_id if name == "intros" else s.lead_channel_id


def private_channels() -> set[int]:
    """Channels whose messages are never mirrored or mined: private intro threads and the lead channel."""
    return {c for c in (channel_setting("intros"), channel_setting("lead")) if c}


def save_channels(intros: int | None, lead: int | None) -> None:
    with session_scope() as session:
        if intros:
            set_setting(session, "discord:intros_channel", str(intros))
        if lead:
            set_setting(session, "discord:lead_channel", str(lead))


def claimed_persona(discord_id: int) -> str | None:
    """The synthetic member this account plays on a demo server, if any."""
    with session_scope() as session:
        return session.scalar(select(Member.name).where(Member.discord_id == discord_id, Member.source == "claimed"))


def join_context(discord_id: int) -> tuple[str | None, dict[str, str]]:
    """What the join flow needs, fetched together: the persona this account plays (if any) and earlier answers."""
    return claimed_persona(discord_id), previous_answers(discord_id)


def previous_answers(discord_id: int) -> dict[str, str]:
    with session_scope() as session:
        member = session.scalar(select(Member).where(Member.discord_id == discord_id))
        if not member:
            return {}
        return {a.question: a.text for a in session.scalars(select(Answer).where(Answer.member_id == member.id))}


def do_onboard(discord_id: int, name: str, answers: dict[str, str], scopes: set[str]) -> dict:
    with session_scope() as session:
        member = get_or_create_discord_member(session, discord_id, name)
        extraction = onboard(session, member, answers, scopes, source="discord:/join")
        log.record(session, "onboard", "ok", f"{len(extraction.facts)} facts via {extraction.extractor}", target=name)
        return {"member_id": member.id, "facts": [(f.kind, f.text) for f in extraction.facts],
                "handles": extraction.handles, "extractor": extraction.extractor, "enrich": "enrich" in scopes}


def do_enrich(member_id: int) -> dict:
    with session_scope() as session:
        return enrich_member(session, session.get(Member, member_id))


def my_profile(discord_id: int) -> dict | None:
    with session_scope() as session:
        member = session.scalar(select(Member).where(Member.discord_id == discord_id))
        if not member:
            return None
        summary = profile_summary(session, member)
        summary["intros"] = [{"week": i.week, "with": (i.member_b if i.member_a_id == member.id else i.member_a).name,
                              "status": i.status, "outcome": i.outcome} for i in cycle.intros_for_member(session, member.id)[:5]]
        return summary


def change_frequency(discord_id: int, frequency: str) -> bool:
    with session_scope() as session:
        member = session.scalar(select(Member).where(Member.discord_id == discord_id))
        if not member:
            return False
        set_frequency(member, frequency)
        log.record(session, "member_frequency", "ok", frequency, target=member.name)
        return True


def forget_me(discord_id: int) -> str | None:
    """'forgotten', 'released <persona>' for an account that only played a demo persona, or None if we hold nothing."""
    with session_scope() as session:
        member = session.scalar(select(Member).where(Member.discord_id == discord_id))
        if not member:
            return None
        if member.source == "claimed":  # the account only played a synthetic member: drop the link, keep the persona
            _release(session, member, db.utcnow())
            return f"released {member.name}"
        forget(session, member, "discord:/forget")
        log.record(session, "member_forget", "ok", "member deleted their data", target=f"member {member.id}")
        return "forgotten"


# ------------------------------------------------------------------ intro cycle

def run_propose(now: datetime, force: bool = False) -> dict:
    with session_scope() as session:
        r = cycle.propose(session, now, force=force)
        return {"week": r.week, "already": r.already, "intros": len(r.intros), "eligible": len(r.eligible),
                "unmatched": len(r.unmatched), "rerank": r.rerank}


def _demo_note(i: Intro) -> str:
    synthetic = [m.name for m in (i.member_a, i.member_b) if m.source == "synthetic"]
    if len(synthetic) == 2:
        return "demo preview: both are synthetic, so it won't be sent"
    if synthetic:
        real = i.member_b.name if synthetic[0] == i.member_a.name else i.member_a.name
        return f"demo: only {real} gets this; {synthetic[0]} is synthetic and won't reply"
    return ""


def proposed_intros(week: str | None = None) -> list[dict]:
    with session_scope() as session:
        q = select(Intro).where(Intro.status == "proposed")
        if week:
            q = q.where(Intro.week == week)
        return [{"id": i.id, "week": i.week, "a": i.member_a.name, "b": i.member_b.name, "score": i.score,
                 "confidence": i.confidence, "why": lead_lines(i.member_a.name, i.member_b.name, i.reasons or []),
                 "writer": (i.copy or {}).get("writer"), "note": _demo_note(i)}
                for i in session.scalars(q.order_by(Intro.score.desc()))]


def veto_intro(intro_id: int, by: str, now: datetime) -> bool:
    with session_scope() as session:
        intro = session.get(Intro, intro_id)
        return bool(intro) and cycle.veto(session, intro, by, now)


def apply_notion_vetoes(now: datetime) -> int:
    if not notion.configured():
        return 0
    with session_scope() as session:
        try:
            return notion.apply_vetoes(session, now)
        except notion.NotionError as exc:
            log.record(session, "notion_veto_check", "failed", str(exc))
            return 0


def due_offers(now: datetime) -> list[int]:
    with session_scope() as session:
        return [i.id for i in cycle.due_to_send(session, now)]


def intro_payload(intro_id: int) -> dict | None:
    """Everything needed to render the offer to both sides."""
    with session_scope() as session:
        i = session.get(Intro, intro_id)
        if not i:
            return None
        sides = {}
        for side, me, other in (("a", i.member_a, i.member_b), ("b", i.member_b, i.member_a)):
            sides[side] = {"discord_id": me.discord_id, "name": me.name, "other_name": other.name,
                           "other_discord_id": other.discord_id, "synthetic": me.source == "synthetic",
                           "lines": render_intro(i.copy, i.reasons, side, other.name)}
        return {"id": i.id, "week": i.week, "status": i.status, "confidence": i.confidence, "score": i.score,
                "opener": (i.copy or {}).get("opener", ""), "sides": sides, "thread_id": i.thread_id}


def mark_offered(intro_id: int, now: datetime, delivery: dict) -> None:
    with session_scope() as session:
        cycle.mark_offered(session, session.get(Intro, intro_id), now, delivery)


def mark_failed(intro_id: int, now: datetime, reason: str) -> None:
    with session_scope() as session:
        cycle.mark_failed(session, session.get(Intro, intro_id), now, reason)


def mark_preview(intro_id: int, now: datetime, reason: str) -> None:
    with session_scope() as session:
        cycle.mark_preview(session, session.get(Intro, intro_id), now, reason)


def respond(intro_id: int, discord_id: int, action: str, now: datetime) -> dict:
    with session_scope() as session:
        intro = session.get(Intro, intro_id)
        member = session.scalar(select(Member).where(Member.discord_id == discord_id))
        if not intro or not member or member.id not in (intro.member_a_id, intro.member_b_id):
            return {"outcome": "not_yours"}
        outcome = cycle.record_response(session, intro, member.id, action, now)
        other = intro.member_b if member.id == intro.member_a_id else intro.member_a
        other_response = intro.b_response if member.id == intro.member_a_id else intro.a_response
        return {"outcome": outcome, "other_name": other.name, "other_discord_id": other.discord_id,
                "other_accepted": other_response == "accept"}


def mark_connected(intro_id: int, thread_id: int | None, now: datetime) -> None:
    with session_scope() as session:
        cycle.mark_connected(session, session.get(Intro, intro_id), thread_id, now)


def expire(now: datetime) -> list[dict]:
    with session_scope() as session:
        return [{"id": i.id, "accepted": [m.discord_id for side, m in (("a", i.member_a), ("b", i.member_b))
                                          if getattr(i, f"{side}_response") == "accept"]}
                for i in cycle.expire_stale(session, now)]


def due_followups(now: datetime) -> list[dict]:
    with session_scope() as session:
        return [{"id": i.id, "thread_id": i.thread_id, "a": i.member_a.discord_id, "b": i.member_b.discord_id}
                for i in cycle.due_followups(session, now)]


def mark_followed_up(intro_id: int, now: datetime) -> None:
    with session_scope() as session:
        cycle.mark_followed_up(session, session.get(Intro, intro_id), now)


def feedback(intro_id: int, discord_id: int, value: str, now: datetime) -> bool:
    with session_scope() as session:
        intro = session.get(Intro, intro_id)
        member = session.scalar(select(Member).where(Member.discord_id == discord_id))
        if not intro or not member or member.id not in (intro.member_a_id, intro.member_b_id):
            return False
        cycle.record_feedback(session, intro, member.id, value, now)
        return True


def due_outcomes(now: datetime) -> list[dict]:
    with session_scope() as session:
        return [{"id": i.id, "thread_id": i.thread_id, "a": i.member_a.discord_id, "b": i.member_b.discord_id}
                for i in cycle.due_outcomes(session, now)]


def finalize(intro_id: int, now: datetime, messages_a: int | None, messages_b: int | None) -> str:
    with session_scope() as session:
        return cycle.finalize(session, session.get(Intro, intro_id), now, messages_a, messages_b)


def propose_due(now: datetime) -> bool:
    with session_scope() as session:
        return cycle.propose_due(session, now) is not None


def digest_due(now: datetime) -> str | None:
    with session_scope() as session:
        return cycle.digest_due(session, now)


def make_digest(week: str, publish: bool = True) -> dict:
    with session_scope() as session:
        d = digest.build(session, week)
        lines = digest.render(d)
        url = None
        if publish and notion.configured():
            try:
                url = notion.publish_digest(session, d, lines)
            except notion.NotionError as exc:
                log.record(session, "notion_digest", "failed", str(exc), target=week)
        set_setting(session, f"cycle:digest:{week}", db.utcnow().isoformat())
        return {"week": week, "lines": lines, "notion_url": url}


# ------------------------------------------------------------------ history + reviver

def last_synced(channel_id: int) -> int | None:
    with session_scope() as session:
        value = get_setting(session, f"hist:{channel_id}")
        return int(value) if value else None


def store_messages(rows: list[dict], cursors: dict[int, int]) -> int:
    if not rows and not cursors:
        return 0
    with session_scope() as session:
        # The demo seeder stored its webhook posts itself, under the synthetic members who "wrote" them.
        seeded = set(json.loads(get_setting(session, SEED_WEBHOOKS_SETTING) or "[]"))
        rows = [{k: v for k, v in r.items() if k != "webhook_id"} for r in rows if r.get("webhook_id") not in seeded]
        if rows:
            if session.bind.dialect.name == "sqlite":
                session.execute(sqlite_insert(Message).values(rows).on_conflict_do_nothing())
            else:
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                session.execute(pg_insert(Message).values(rows).on_conflict_do_nothing())
        for channel_id, last_id in cursors.items():
            set_setting(session, f"hist:{channel_id}", str(last_id))
        rebuild_interactions(session)
    return len(rows)


def revive_scan(now: datetime) -> list[dict]:
    with session_scope() as session:
        rows = reviver.scan(session, now)
        return [revival_payload(r, session) for r in rows]


def pending_revivals(exclude: set[int] = frozenset()) -> list[dict]:
    """Drafts from earlier scans still waiting for the lead's Post/Skip."""
    with session_scope() as session:
        return [revival_payload(r, session) for r in session.scalars(select(Revival).where(Revival.status == "drafted")
                                                                      .order_by(Revival.detected_at)) if r.id not in exclude]


def revival_payload(r: Revival, session=None) -> dict:
    text, mentions = reviver.render(r, session) if r.draft else ("", [])
    return {"id": r.id, "title": r.title, "status": r.status, "rule": r.rule, "channel_id": r.channel_id,
            "thread_id": r.thread_id, "anchor_message_id": r.anchor_message_id, "text": text, "mentions": mentions,
            "reason": r.failure_reason or (r.judge or {}).get("reason", ""), "score": (r.judge or {}).get("score"),
            "silent_days": (r.judge or {}).get("silent_days"), "channel_name": (r.judge or {}).get("channel_name")}


def revivals_to_post(now: datetime) -> list[dict]:
    with session_scope() as session:
        return [revival_payload(r, session) for r in reviver.to_post(session, now)]


def get_revival(revival_id: int) -> dict | None:
    with session_scope() as session:
        r = session.get(Revival, revival_id)
        return revival_payload(r, session) if r else None


def revival_posted(revival_id: int, message_id: int, now: datetime) -> None:
    with session_scope() as session:
        reviver.mark_posted(session, session.get(Revival, revival_id), message_id, now)


def revival_failed(revival_id: int, reason: str, now: datetime) -> None:
    with session_scope() as session:
        reviver.mark_failed(session, session.get(Revival, revival_id), reason, now)


def revival_skip(revival_id: int, by: str) -> None:
    with session_scope() as session:
        r = session.get(Revival, revival_id)
        if r and r.status == "drafted":
            r.status, r.failure_reason = "skipped", f"skipped by {by}"
            log.record(session, "revive_skip", "skipped", f"skipped by {by}", target=r.title[:200])


def revive_evaluate(now: datetime) -> int:
    with session_scope() as session:
        return len(reviver.evaluate(session, now))


# ------------------------------------------------------------------ demo personas

def persona_names(typed: str = "", claimed: bool = False, limit: int = 25) -> list[str]:
    """Synthetic members (or, with claimed=True, the ones a real account plays) whose name contains `typed`."""
    with session_scope() as session:
        q = select(Member.name).where(Member.source == ("claimed" if claimed else "synthetic"), Member.status == "active")
        return [n for n in session.scalars(q.order_by(Member.name)) if typed.lower() in n.lower()][:limit]


def _find_persona(session, name: str) -> Member | None:
    return session.scalar(select(Member).where(func.lower(Member.name) == name.strip().lower(),
                                               Member.source.in_(("synthetic", "claimed"))))


def claim_persona(name: str, discord_id: int, display_name: str, now: datetime) -> dict:
    """Demo servers: a real Discord account plays a synthetic member, so that member's intros, DMs and pings reach a
    real person. The matcher is untouched: it still decides who meets whom from the profiles alone."""
    with session_scope() as session:
        persona = _find_persona(session, name)
        if persona is None:
            return {"ok": False, "error": f"There's no synthetic member called {name}."}
        if persona.source == "claimed":
            return {"ok": False, "error": f"{persona.name} is already played by <@{persona.discord_id}>. Run `/kym unclaim` first."}
        existing = session.scalar(select(Member).where(Member.discord_id == discord_id))
        if existing is not None:
            if existing.source == "claimed":
                return {"ok": False, "error": f"That account already plays {existing.name}."}
            if existing.status == "active":
                return {"ok": False, "error": f"That account has its own profile ({existing.name}). "
                                              "Run `/forget` with it first, or pick another account."}
            existing.discord_id = None  # a forgotten profile: free the Discord id, keep the tombstone
            session.flush()
        synthetic_id = persona.discord_id
        moved = list(session.scalars(select(Message.id).where(Message.author_id == synthetic_id)))
        persona.discord_id, persona.source = discord_id, "claimed"
        persona.meta = {**(persona.meta or {}), "synthetic_discord_id": synthetic_id, "played_by": display_name,
                        "claimed_message_ids": moved}
        session.flush()
        if moved:
            session.execute(update(Message).where(Message.id.in_(moved)).values(author_id=discord_id))
        log.record(session, "persona_claim", "ok", f"played by {display_name}", target=persona.name, at=now)
        return {"ok": True, "name": persona.name, "facts": [(f.kind, f.text) for f in persona.facts][:6]}


def _release(session, persona: Member, now: datetime) -> None:
    meta = dict(persona.meta or {})
    synthetic_id, moved = meta.pop("synthetic_discord_id", None), meta.pop("claimed_message_ids", [])
    player = meta.pop("played_by", "")
    persona.discord_id, persona.source, persona.meta = synthetic_id, "synthetic", meta
    session.flush()
    if moved:
        session.execute(update(Message).where(Message.id.in_(moved)).values(author_id=synthetic_id))
    log.record(session, "persona_release", "ok", f"no longer played by {player}", target=persona.name, at=now)


def release_persona(name: str, now: datetime) -> dict:
    with session_scope() as session:
        persona = _find_persona(session, name)
        if persona is None or persona.source != "claimed":
            return {"ok": False, "error": f"Nobody plays {name}."}
        _release(session, persona, now)
        return {"ok": True, "name": persona.name}


# ------------------------------------------------------------------ speakers, stats, notion

def speakers(topic: str, publish: bool = True) -> dict:
    with session_scope() as session:
        result = find_speakers(session, topic)
        if publish and notion.configured() and result["shortlist"]:
            try:
                result["notion_url"] = notion.publish_speakers(session, result)
            except notion.NotionError as exc:
                result["notes"].append(f"Notion publish failed: {exc}")
        return result


def notion_sync() -> dict:
    with session_scope() as session:
        return notion.sync(session)


def recent_log(limit: int = 15) -> list[dict]:
    from ..models import Attempt

    with session_scope() as session:
        return [{"at": a.at, "kind": a.kind, "status": a.status, "target": a.target, "detail": a.detail}
                for a in session.scalars(select(Attempt).order_by(Attempt.at.desc()).limit(limit))]


QUESTION_KEYS = [q.key for q in QUESTIONS]
