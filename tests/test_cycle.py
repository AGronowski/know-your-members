from datetime import timedelta

from kym import cycle
from kym.cycle import propose

from .conftest import MONDAY

TUESDAY = MONDAY + timedelta(days=1)


def _first(demo):
    r = propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    return r.intros[0]


def test_not_sent_before_the_veto_window_closes(demo):
    propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    assert cycle.due_to_send(demo, MONDAY + timedelta(hours=1)) == []
    assert cycle.due_to_send(demo, TUESDAY)


def test_veto_stops_an_intro(demo):
    intro = _first(demo)
    assert cycle.veto(demo, intro, "lead", MONDAY)
    assert intro not in cycle.due_to_send(demo, TUESDAY)
    assert not cycle.veto(demo, intro, "lead", MONDAY)


def test_double_opt_in_then_follow_up_and_outcome(demo):
    intro = _first(demo)
    cycle.mark_offered(demo, intro, TUESDAY, {"test": True})
    assert cycle.record_response(demo, intro, intro.member_a_id, "accept", TUESDAY) == "waiting"
    assert cycle.record_response(demo, intro, intro.member_b_id, "accept", TUESDAY) == "connect"
    cycle.mark_connected(demo, intro, 123, TUESDAY)
    assert cycle.due_followups(demo, TUESDAY + timedelta(days=1)) == []
    assert intro in cycle.due_followups(demo, TUESDAY + timedelta(days=5))
    cycle.mark_followed_up(demo, intro, TUESDAY + timedelta(days=5))
    cycle.record_feedback(demo, intro, intro.member_a_id, "useful", TUESDAY + timedelta(days=5))
    assert intro in cycle.due_outcomes(demo, TUESDAY + timedelta(days=8))
    assert cycle.finalize(demo, intro, TUESDAY + timedelta(days=8), 4, 3) == "useful"
    assert intro.status == "completed"


def test_decline_closes_and_pause_stops_future_intros(demo):
    intro = _first(demo)
    cycle.mark_offered(demo, intro, TUESDAY, {})
    assert cycle.record_response(demo, intro, intro.member_a_id, "pause", TUESDAY) == "declined"
    assert intro.member_a.intro_frequency == "paused"
    assert cycle.record_response(demo, intro, intro.member_b_id, "accept", TUESDAY) == "closed"


def test_silence_expires_and_lowers_responsiveness(demo):
    intro = _first(demo)
    before = intro.member_b.responsiveness
    cycle.mark_offered(demo, intro, TUESDAY, {})
    cycle.record_response(demo, intro, intro.member_a_id, "accept", TUESDAY)
    expired = cycle.expire_stale(demo, TUESDAY + timedelta(days=4))
    assert intro in expired and intro.status == "expired"
    assert intro.member_b.responsiveness < before


def test_thread_activity_beats_a_contradicting_vote_and_the_conflict_is_logged(demo):
    from sqlalchemy import select

    from kym.models import Attempt

    intro = _first(demo)
    cycle.mark_offered(demo, intro, TUESDAY, {})
    cycle.record_response(demo, intro, intro.member_a_id, "accept", TUESDAY)
    cycle.record_response(demo, intro, intro.member_b_id, "accept", TUESDAY)
    cycle.mark_connected(demo, intro, 1, TUESDAY)
    cycle.record_feedback(demo, intro, intro.member_a_id, "didnt_happen", TUESDAY + timedelta(days=5))
    assert cycle.finalize(demo, intro, TUESDAY + timedelta(days=8), 9, 7) == "talked"
    logged = demo.scalars(select(Attempt).where(Attempt.kind == "intro_outcome")).all()[-1]
    assert "didn't happen" in logged.detail and "both wrote" in logged.detail


def test_no_show_stays_a_no_show(demo):
    intro = _first(demo)
    cycle.mark_offered(demo, intro, TUESDAY, {})
    cycle.record_response(demo, intro, intro.member_a_id, "accept", TUESDAY)
    cycle.record_response(demo, intro, intro.member_b_id, "accept", TUESDAY)
    cycle.mark_connected(demo, intro, 1, TUESDAY)
    cycle.record_feedback(demo, intro, intro.member_b_id, "didnt_happen", TUESDAY + timedelta(days=5))
    assert cycle.finalize(demo, intro, TUESDAY + timedelta(days=8), 1, 0) == "didnt_happen"


def test_outcome_from_messages_when_nobody_answers_the_follow_up(demo):
    intro = _first(demo)
    cycle.mark_offered(demo, intro, TUESDAY, {})
    cycle.record_response(demo, intro, intro.member_a_id, "accept", TUESDAY)
    cycle.record_response(demo, intro, intro.member_b_id, "accept", TUESDAY)
    cycle.mark_connected(demo, intro, 1, TUESDAY)
    assert cycle.finalize(demo, intro, TUESDAY + timedelta(days=8), 0, 0) == "silent"


def test_simulated_month_keeps_invariants(tmp_path):
    from kym.simulate import simulate

    r = simulate(weeks=3, db_path=tmp_path / "sim.db")
    assert r["invariants"] == {"repeat_pairs": 0, "members_with_two_intros_in_a_week": 0, "intros_without_consent": 0}
    assert r["late_joiner"]["eligible_in_first_week"]  # joined mid-month, entered the cycle with no organiser
    assert r["graph_after"]["share_connected"] > r["graph_before"]["share_connected"]
    from sqlalchemy import select

    from kym import db
    from kym.models import Intro

    with db.session_scope() as s:  # simulated feedback agrees with what happened in the thread
        for i in s.scalars(select(Intro).where(Intro.status == "completed")):
            if i.outcome == "didnt_happen":
                assert not ((i.messages_a or 0) > 0 and (i.messages_b or 0) > 0)
            if i.outcome in ("useful", "meh"):
                assert (i.messages_a or 0) > 0 and (i.messages_b or 0) > 0
