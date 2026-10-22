"""The weekly intro cycle as an idempotent state machine.

Monday: propose (lead can veto in Notion or Discord) -> Tuesday: offer to both members (double opt-in) ->
both say yes: private thread -> day 4: follow-up question -> day 7: outcome from feedback + message counts.
Every step checks the database before acting, so a restarted bot (or one that missed a day) catches up without
double-sending. Discord side effects live in the bot; this module only moves state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from . import clock, log
from .config import get_settings
from .db import get_setting, set_setting
from .graph import build_graph, communities, interaction_weights, pair
from .matching import MatchConfig, assign, confidence_label, explain, load_profiles, rerank, score_pairs
from .models import Fact, Intro, Member
from .writer import template_copy, write_intro

ACTIVE_STATUSES = ("proposed", "offered", "connected")
OPEN_OFFER = ("offered",)


@dataclass
class ProposeResult:
    week: str
    intros: list[Intro] = field(default_factory=list)
    eligible: list[int] = field(default_factory=list)
    unmatched: dict[int, str] = field(default_factory=dict)
    rerank: dict | None = None
    already: bool = False


def _eligible(session: Session, week: str, include_synthetic: bool) -> tuple[list[int], dict[int, str]]:
    """Members who can get an intro this week, plus why the others can't."""
    skipped: dict[int, str] = {}
    members = session.scalars(select(Member).where(Member.status == "active")).all()
    if not include_synthetic:
        members = [m for m in members if m.source != "synthetic"]
    with_facts = {mid for (mid,) in session.execute(select(Fact.member_id).where(Fact.kind.in_(("wants", "offers"))).distinct())}
    busy = {mid for i in session.scalars(select(Intro).where(Intro.status.in_(OPEN_OFFER) | (Intro.week == week)))
            for mid in (i.member_a_id, i.member_b_id)}
    last_week = clock.previous_week(week)
    had_last_week = {mid for i in session.scalars(select(Intro).where(Intro.week == last_week,
                                                                    Intro.status.not_in(("vetoed", "failed", "preview"))))
                     for mid in (i.member_a_id, i.member_b_id)}
    eligible = []
    for m in members:
        if not m.allows("intro"):
            skipped[m.id] = "no intro consent"
        elif m.intro_frequency == "paused":
            skipped[m.id] = "paused"
        elif m.intro_frequency == "fortnightly" and m.id in had_last_week:
            skipped[m.id] = "fortnightly: had one last week"
        elif not m.discord_id:
            skipped[m.id] = "not on Discord yet"
        elif m.id not in with_facts:
            skipped[m.id] = "no answers yet"
        elif m.id in busy:
            skipped[m.id] = "intro already in progress"
        else:
            eligible.append(m.id)
    return eligible, skipped


def propose(session: Session, now: datetime, *, week: str | None = None, force: bool = False,
            use_llm: bool | None = None, cfg: MatchConfig | None = None, include_synthetic: bool | None = None) -> ProposeResult:
    """Match this week's eligible members and draft their intros (status 'proposed')."""
    s = get_settings()
    week = week or clock.week_id(now)
    marker = f"cycle:proposed:{week}"
    if get_setting(session, marker) and not force:
        return ProposeResult(week=week, already=True)
    if force:  # redo whatever hasn't gone out yet; sent intros stay
        for intro in session.scalars(select(Intro).where(Intro.week == week, Intro.status.in_(("proposed", "preview")))):
            session.delete(intro)
        session.flush()
    eligible, skipped = _eligible(session, week, s.include_synthetic if include_synthetic is None else include_synthetic)
    result = ProposeResult(week=week, eligible=eligible)
    if len(eligible) < 2:
        set_setting(session, marker, now.isoformat())
        log.record(session, "intro_propose", "skipped", f"{len(eligible)} eligible member(s); need at least 2", target=week)
        return result

    excluded = {pair(i.member_a_id, i.member_b_id) for i in session.scalars(select(Intro))}
    profiles = load_profiles(session, eligible)
    g = build_graph(session)
    carry = set(json.loads(get_setting(session, f"cycle:unmatched:{clock.previous_week(week)}") or "[]"))
    candidates, index, F = score_pairs(session, profiles, known=interaction_weights(session), excluded=excluded,
                                       community=communities(g), carry_over=carry & set(eligible), cfg=cfg)
    if (s.llm_rerank if use_llm is None else use_llm) and candidates:
        result.rerank = rerank(session, candidates, profiles, cfg or MatchConfig())
        if result.rerank.get("status") == "failed":
            log.record(session, "intro_rerank", "failed", "; ".join(result.rerank.get("failures", [])), target=week)
    chosen = assign(candidates)
    matched = set()
    for c in chosen:
        reasons = explain(index, F, profiles, c.a, c.b, (cfg or MatchConfig()).min_reason)
        if use_llm is False:
            copy = {**template_copy(profiles[c.a].name, profiles[c.b].name, reasons), "writer": "template", "dropped": []}
        else:
            copy = write_intro(session, profiles[c.a].name, profiles[c.b].name, reasons)
        intro = Intro(week=week, member_a_id=c.a, member_b_id=c.b, score=c.score, confidence=confidence_label(c.score),
                      breakdown={**c.breakdown, **({"llm": c.llm} if c.llm else {})}, reasons=reasons, copy=copy,
                      status="proposed", created_at=now, updated_at=now)
        session.add(intro)
        result.intros.append(intro)
        matched |= {c.a, c.b}
        if copy.get("dropped"):
            log.record(session, "intro_writer", "failed", f"dropped {len(copy['dropped'])} unverifiable claim(s)",
                       target=f"{profiles[c.a].name} x {profiles[c.b].name}", dropped=copy["dropped"], at=now)
    candidate_members = {m for c in candidates for m in (c.a, c.b)}
    for mid in eligible:
        if mid not in matched:
            result.unmatched[mid] = ("no strong enough match this week" if mid not in candidate_members
                                     else "best matches went to others in the global assignment")
    set_setting(session, f"cycle:unmatched:{week}", json.dumps(sorted(result.unmatched)))
    set_setting(session, marker, now.isoformat())
    session.flush()
    log.record(session, "intro_propose", "ok",
               f"{len(result.intros)} intros for {len(eligible)} eligible members; {len(result.unmatched)} unmatched",
               target=week, unmatched={str(k): v for k, v in result.unmatched.items()},
               skipped={str(k): v for k, v in skipped.items()}, at=now)
    return result


def veto(session: Session, intro: Intro, by: str, now: datetime) -> bool:
    if intro.status != "proposed":
        return False
    intro.status, intro.vetoed, intro.updated_at = "vetoed", True, now
    log.record(session, "intro_veto", "ok", f"vetoed by {by}", target=str(intro.id), at=now)
    return True


def due_to_send(session: Session, now: datetime) -> list[Intro]:
    s = get_settings()
    out = []
    for intro in session.scalars(select(Intro).where(Intro.status == "proposed").order_by(Intro.id)):
        if now >= clock.at(intro.week, s.send_weekday, s.send_hour):
            out.append(intro)
    return out


def mark_offered(session: Session, intro: Intro, now: datetime, delivery: dict) -> None:
    intro.status, intro.offered_at, intro.updated_at = "offered", now, now
    intro.copy = {**intro.copy, "delivery": delivery}
    log.record(session, "intro_send", "ok", json.dumps(delivery)[:500], target=str(intro.id), at=now)


def mark_preview(session: Session, intro: Intro, now: datetime, reason: str) -> None:
    """Demo servers only: an intro between two synthetic members is shown to the lead but never sent."""
    intro.status, intro.failure, intro.updated_at = "preview", reason, now
    log.record(session, "intro_send", "skipped", reason, target=str(intro.id), at=now)


def mark_failed(session: Session, intro: Intro, now: datetime, reason: str) -> None:
    intro.status, intro.failure, intro.updated_at = "failed", reason, now
    log.record(session, "intro_send", "failed", reason, target=str(intro.id), at=now)


def _nudge_responsiveness(member: Member, responded: bool) -> None:
    target = 1.0 if responded else 0.0
    member.responsiveness = round(0.8 * (member.responsiveness or 0.7) + 0.2 * target, 3)


def record_response(session: Session, intro: Intro, member_id: int, response: str, now: datetime) -> str:
    """Returns 'connect' (both said yes), 'waiting', 'declined' or 'closed'."""
    if response not in ("accept", "decline", "pause"):
        raise ValueError(response)
    if intro.status != "offered":
        return "closed"
    side = intro.side(member_id)
    member = session.get(Member, member_id)
    setattr(intro, f"{side}_response", response)
    setattr(intro, f"{side}_responded_at", now)
    intro.updated_at = now
    _nudge_responsiveness(member, True)
    if response == "pause":
        member.intro_frequency = "paused"
    log.record(session, "intro_response", "ok", f"{member.name}: {response}", target=str(intro.id), at=now)
    if response in ("decline", "pause"):
        intro.status = "declined"
        return "declined"
    if intro.a_response == "accept" and intro.b_response == "accept":
        return "connect"
    return "waiting"


def mark_connected(session: Session, intro: Intro, thread_id: int | None, now: datetime) -> None:
    intro.status, intro.thread_id, intro.connected_at, intro.updated_at = "connected", thread_id, now, now
    log.record(session, "intro_connect", "ok", f"thread {thread_id}", target=str(intro.id), at=now)


def expire_stale(session: Session, now: datetime) -> list[Intro]:
    ttl = timedelta(hours=get_settings().response_ttl_hours)
    expired = []
    for intro in session.scalars(select(Intro).where(Intro.status == "offered")):
        if intro.offered_at and now - intro.offered_at >= ttl:
            intro.status, intro.updated_at = "expired", now
            for side, mid in (("a", intro.member_a_id), ("b", intro.member_b_id)):
                if getattr(intro, f"{side}_response") is None:
                    _nudge_responsiveness(session.get(Member, mid), False)
            silent = [m for s_, m in (("a", intro.member_a), ("b", intro.member_b)) if getattr(intro, f"{s_}_response") is None]
            log.record(session, "intro_expire", "failed", "no reply from " + ", ".join(m.name for m in silent),
                       target=str(intro.id), at=now)
            expired.append(intro)
    return expired


def due_followups(session: Session, now: datetime) -> list[Intro]:
    after = timedelta(hours=get_settings().followup_after_hours)
    return [i for i in session.scalars(select(Intro).where(Intro.status == "connected", Intro.followed_up_at.is_(None)))
            if i.connected_at and now - i.connected_at >= after]


def mark_followed_up(session: Session, intro: Intro, now: datetime) -> None:
    intro.followed_up_at, intro.updated_at = now, now
    log.record(session, "intro_followup", "ok", "asked whether they connected", target=str(intro.id), at=now)


def record_feedback(session: Session, intro: Intro, member_id: int, value: str, now: datetime) -> None:
    if value not in ("useful", "meh", "didnt_happen"):
        raise ValueError(value)
    intro.side(member_id)
    intro.feedback = {**(intro.feedback or {}), str(member_id): value}
    intro.updated_at = now
    log.record(session, "intro_feedback", "ok", value, target=str(intro.id), at=now)


def due_outcomes(session: Session, now: datetime) -> list[Intro]:
    after = timedelta(hours=get_settings().outcome_after_hours)
    return [i for i in session.scalars(select(Intro).where(Intro.status == "connected"))
            if i.connected_at and now - i.connected_at >= after]


def finalize(session: Session, intro: Intro, now: datetime, messages_a: int | None, messages_b: int | None) -> str:
    """Outcome from what members said, falling back to whether both actually wrote in the thread."""
    intro.messages_a, intro.messages_b = messages_a, messages_b
    votes = list((intro.feedback or {}).values())
    both_wrote = (messages_a or 0) > 0 and (messages_b or 0) > 0
    note = ""
    if "useful" in votes:
        outcome = "useful"
    elif "meh" in votes:
        outcome = "meh"
    elif "didnt_happen" in votes and both_wrote:
        outcome, note = "talked", "; a member answered 'didn't happen' but both wrote in the thread"
    elif "didnt_happen" in votes:
        outcome = "didnt_happen"
    elif both_wrote:
        outcome = "talked"
    elif (messages_a or 0) + (messages_b or 0) > 0:
        outcome = "one_sided"
    else:
        outcome = "silent"
    intro.outcome, intro.status, intro.updated_at = outcome, "completed", now
    status = "ok" if outcome in ("useful", "talked", "meh") else "failed"
    log.record(session, "intro_outcome", status, f"{outcome} (messages a={messages_a}, b={messages_b}){note}",
               target=str(intro.id), at=now)
    return outcome


def digest_due(session: Session, now: datetime) -> str | None:
    s = get_settings()
    week = clock.week_id(now)
    if now >= clock.at(week, s.digest_weekday, s.digest_hour) and not get_setting(session, f"cycle:digest:{week}"):
        return week
    return None


def propose_due(session: Session, now: datetime) -> str | None:
    s = get_settings()
    week = clock.week_id(now)
    if now >= clock.at(week, s.propose_weekday, s.propose_hour) and not get_setting(session, f"cycle:proposed:{week}"):
        return week
    return None


def intros_for_member(session: Session, member_id: int) -> list[Intro]:
    return session.scalars(select(Intro).where(or_(Intro.member_a_id == member_id, Intro.member_b_id == member_id))
                           .order_by(Intro.id.desc())).all()
