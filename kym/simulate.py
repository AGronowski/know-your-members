"""Run the real weekly cycle for N weeks on the synthetic community, with simulated members.

This is the "does it still work a month later without the organiser?" check: same code paths as the bot (propose,
offer, double opt-in, expiry, follow-up, outcome), a fake clock, and invariants verified at the end.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from . import clock, cycle, db
from .config import DATA_DIR, get_settings
from .graph import build_graph, metrics
from .importer import seed_demo
from .members import onboard
from .models import CONSENT_SCOPES, Attempt, Intro, Member

LATE_JOINER = {
    "name": "Noor Hassan",
    "answers": {"building": "Voice assistants for blind and partially sighted users",
                "offers": "Accessibility testing; screen reader UX",
                "wants": "Natural-sounding text-to-speech; a mentor in AI engineering",
                "speak": "Yes: accessible voice interfaces"},
}


def _fresh_db(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    db.configure(f"sqlite:///{path}")
    db.init_db()


def simulate(weeks: int = 4, seed: int = 7, start: datetime | None = None, db_path: Path | None = None,
             use_llm: bool = False, late_joiner_week: int | None = 3) -> dict:
    s = get_settings()
    rng = random.Random(seed)
    start = start or datetime(2026, 10, 26, 8, 0, tzinfo=timezone.utc)
    _fresh_db(db_path or DATA_DIR / "sim.db")
    with db.session_scope() as session:
        seed_demo(session, now=start - timedelta(days=1))
        before = metrics(build_graph(session))
    week = clock.week_id(start)
    rows = []
    late_joiner_eligible = None
    for n in range(1, weeks + 1):
        t_propose = clock.at(week, s.propose_weekday, s.propose_hour)
        t_send = clock.at(week, s.send_weekday, s.send_hour)
        joined = None
        if late_joiner_week and n == late_joiner_week:
            with db.session_scope() as session:
                m = Member(discord_id=990_000_000_000_000_001, name=LATE_JOINER["name"], source="synthetic",
                           status="active", handles={}, consent={}, meta={}, joined_at=t_propose - timedelta(days=2))
                session.add(m)
                session.flush()
                onboard(session, m, LATE_JOINER["answers"], set(CONSENT_SCOPES), "sim:late-joiner", use_llm=use_llm)
                joined = m.name
        with db.session_scope() as session:
            proposed = cycle.propose(session, t_propose, week=week, use_llm=use_llm, include_synthetic=True)
            eligible, unmatched = len(proposed.eligible), len(proposed.unmatched)
            if joined:
                joiner_id = session.scalar(select(Member.id).where(Member.name == joined))
                late_joiner_eligible = joiner_id in proposed.eligible
        with db.session_scope() as session:
            for intro in cycle.due_to_send(session, t_send):
                cycle.mark_offered(session, intro, t_send, {"simulated": True})
        with db.session_scope() as session:
            for intro in session.scalars(select(Intro).where(Intro.week == week, Intro.status == "offered")):
                for member in (intro.member_a, intro.member_b):
                    if intro.status != "offered" or rng.random() > member.responsiveness:
                        continue  # this member never answers
                    p_accept = min(0.95, 0.4 + 0.6 * intro.score)
                    roll = rng.random()
                    response = "accept" if roll < p_accept else ("pause" if roll > 0.97 else "decline")
                    when = t_send + timedelta(hours=rng.uniform(1, 48))
                    if cycle.record_response(session, intro, member.id, response, when) == "connect":
                        cycle.mark_connected(session, intro, None, when)
        with db.session_scope() as session:
            cycle.expire_stale(session, t_send + timedelta(hours=s.response_ttl_hours + 1))
        t_follow = t_send + timedelta(hours=s.followup_after_hours + 49)
        t_outcome = t_send + timedelta(hours=s.outcome_after_hours + 49)
        talked: dict[int, bool] = {}
        with db.session_scope() as session:
            for intro in cycle.due_followups(session, t_follow):
                cycle.mark_followed_up(session, intro, t_follow)
                talked[intro.id] = rng.random() < 0.4 + 0.5 * intro.score  # did this pair actually talk?
                for member in (intro.member_a, intro.member_b):
                    if rng.random() < member.responsiveness * 0.8:
                        if talked[intro.id]:
                            value = "useful" if rng.random() < 0.3 + 0.6 * intro.score else "meh"
                        else:
                            value = "didnt_happen"
                        cycle.record_feedback(session, intro, member.id, value, t_follow)
            for intro in cycle.due_outcomes(session, t_outcome):
                if talked.get(intro.id, False):
                    messages = (rng.randint(2, 9), rng.randint(2, 9))
                else:
                    messages = (rng.randint(0, 1), 0)  # at most one hopeful message
                cycle.finalize(session, intro, t_outcome, *messages)
        with db.session_scope() as session:
            intros = session.scalars(select(Intro).where(Intro.week == week)).all()
            status = Counter(i.status for i in intros)
            outcomes = Counter(i.outcome for i in intros if i.outcome)
            rows.append({
                "week": week, "eligible": eligible, "intros": len(intros), "unmatched": unmatched,
                "connected": status["completed"] + status["connected"], "declined": status["declined"],
                "no_reply": status["expired"], "useful": outcomes["useful"], "talked": outcomes["talked"],
                "meh": outcomes["meh"], "silent_or_no_show": outcomes["silent"] + outcomes["didnt_happen"] + outcomes["one_sided"],
                "avg_score": round(sum(i.score for i in intros) / len(intros), 3) if intros else 0,
                "joined": joined,
            })
        week = clock.next_week(week)

    with db.session_scope() as session:
        all_intros = session.scalars(select(Intro)).all()
        members = {m.id: m for m in session.scalars(select(Member))}
        pairs = Counter(frozenset((i.member_a_id, i.member_b_id)) for i in all_intros)
        per_week = Counter((i.week, mid) for i in all_intros for mid in (i.member_a_id, i.member_b_id))
        no_consent = [i.id for i in all_intros if not members[i.member_a_id].consent.get("intro") or not members[i.member_b_id].consent.get("intro")]
        introduced = {mid for i in all_intros if i.status != "vetoed" for mid in (i.member_a_id, i.member_b_id)}
        opted_in = [m for m in members.values() if m.consent.get("intro")]
        failures = session.scalars(select(Attempt).where(Attempt.status == "failed").order_by(Attempt.at)).all()
        late = next((m for m in members.values() if m.name == LATE_JOINER["name"]), None)
        result = {
            "weeks": rows,
            "invariants": {
                "repeat_pairs": sum(1 for c in pairs.values() if c > 1),
                "members_with_two_intros_in_a_week": sum(1 for c in per_week.values() if c > 1),
                "intros_without_consent": len(no_consent),
            },
            "coverage": {"opted_in": len(opted_in), "introduced_at_least_once": len(introduced & {m.id for m in opted_in}),
                         "paused_by_members": sum(1 for m in members.values() if m.intro_frequency == "paused")},
            "late_joiner": {"name": late.name, "eligible_in_first_week": late_joiner_eligible,
                            "intros": sum(1 for i in all_intros if late.id in (i.member_a_id, i.member_b_id))} if late else None,
            "graph_before": before,
            "graph_after": metrics(build_graph(session)),
            "failures": Counter(f.kind for f in failures),
            "failure_examples": [f"{f.kind}: {f.detail}" for f in failures[:6]],
        }
    return result
