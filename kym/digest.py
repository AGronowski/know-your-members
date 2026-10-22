"""The lead's weekly digest: what the bot did, what worked, what failed and why."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import clock
from .graph import build_graph, metrics
from .models import Attempt, ConsentEvent, Intro, Member, Revival, SpeakerSearch


def build(session: Session, week: str) -> dict:
    start = clock.week_start(week)
    end = start + timedelta(days=7)
    this_week = session.scalars(select(Intro).where(Intro.week == week)).all()
    last_week = session.scalars(select(Intro).where(Intro.week == clock.previous_week(week))).all()
    revivals = session.scalars(select(Revival).where(Revival.detected_at >= start, Revival.detected_at < end)).all()
    failures = session.scalars(select(Attempt).where(Attempt.at >= start, Attempt.at < end, Attempt.status == "failed")
                               .order_by(Attempt.at)).all()
    members = session.scalars(select(Member)).all()
    joined = [m for m in members if m.onboarded_at and start <= m.onboarded_at < end]
    opt_outs = session.scalars(select(ConsentEvent).where(ConsentEvent.at >= start, ConsentEvent.at < end,
                                                          ConsentEvent.granted.is_(False), ConsentEvent.scope == "intro")).all()
    all_intros = session.scalars(select(Intro).where(Intro.status.not_in(("vetoed",)))).all()
    opted_in = {m.id for m in members if m.allows("intro")}
    introduced = {mid for i in all_intros for mid in (i.member_a_id, i.member_b_id)} & opted_in
    searches = session.scalars(select(SpeakerSearch).where(SpeakerSearch.created_at >= start, SpeakerSearch.created_at < end)).all()
    return {
        "week": week,
        "intros": Counter(i.status for i in this_week),
        "intro_count": len(this_week),
        "last_week_outcomes": Counter(i.outcome or i.status for i in last_week),
        "examples": [{"pair": f"{i.member_a.name} × {i.member_b.name}", "status": i.status, "confidence": i.confidence,
                      "why": (i.copy or {}).get("to_a", [{}])[0].get("text", "")} for i in this_week[:5]],
        "revivals": Counter(r.status for r in revivals),
        "revival_examples": [{"title": r.title, "status": r.status, "why": r.failure_reason or (r.judge or {}).get("reason", "")}
                             for r in revivals[:5]],
        "joined": [m.name for m in joined],
        "opted_out": len(opt_outs),
        "coverage": {"opted_in": len(opted_in), "introduced_ever": len(introduced),
                     "prospects": sum(1 for m in members if m.status == "prospect")},
        "failures": Counter(f.kind for f in failures),
        "failure_examples": [f"{f.kind} ({f.target}): {f.detail}"[:200] for f in failures[:8]],
        "speaker_searches": [s.topic for s in searches],
        "graph": metrics(build_graph(session)),
    }


def render(d: dict) -> list[str]:
    intros = d["intros"]
    lines = [
        f"Week {d['week']}",
        f"• Intros: {d['intro_count']} proposed; {intros.get('offered', 0)} waiting on replies, "
        f"{intros.get('connected', 0) + intros.get('completed', 0)} connected, {intros.get('declined', 0)} declined, "
        f"{intros.get('expired', 0)} got no reply, {intros.get('vetoed', 0)} vetoed, {intros.get('failed', 0)} failed to send"
        + (f", {intros['preview']} between synthetic demo members shown as previews only" if intros.get("preview") else ""),
    ]
    if d["last_week_outcomes"]:
        lines.append("• Last week's intros ended as: " + ", ".join(f"{k} {v}" for k, v in d["last_week_outcomes"].most_common()))
    for ex in d["examples"]:
        lines.append(f"  - {ex['pair']} ({ex['confidence']}, {ex['status']}): {ex['why']}")
    if d["revivals"]:
        lines.append("• Thread reviver: " + ", ".join(f"{k} {v}" for k, v in d["revivals"].most_common()))
        for ex in d["revival_examples"]:
            lines.append(f"  - \"{ex['title'][:60]}\": {ex['status']}" + (f" ({ex['why'][:100]})" if ex["why"] else ""))
    c = d["coverage"]
    lines.append(f"• Members: {c['opted_in']} opted in, {c['introduced_ever']} introduced at least once, "
                 f"{c['prospects']} prospects not yet opted in; joined this week: {', '.join(d['joined']) or 'nobody'}; "
                 f"opted out of intros: {d['opted_out']}")
    g = d["graph"]
    if g.get("members"):
        lines.append(f"• Network: {g['connections']} known connections ({g.get('intro_connections', 0)} made by intros), "
                     f"{g['isolated_members']} members with no connection yet, density {g['density']}")
    if d["speaker_searches"]:
        lines.append("• Speaker searches: " + ", ".join(d["speaker_searches"]))
    if d["failures"]:
        lines.append("• What failed: " + ", ".join(f"{k} ×{v}" for k, v in d["failures"].most_common()))
        lines += [f"  - {f}" for f in d["failure_examples"]]
    else:
        lines.append("• Nothing failed this week.")
    return lines
