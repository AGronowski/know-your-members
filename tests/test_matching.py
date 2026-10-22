from collections import Counter

from sqlalchemy import select

from kym.cycle import propose
from kym.models import Member

from .conftest import MONDAY


def _names(session):
    return {m.id: m.name for m in session.scalars(select(Member))}


def test_complementary_pairs_win(demo):
    r = propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    names = _names(demo)
    pairs = {frozenset((names[i.member_a_id], names[i.member_b_id])) for i in r.intros}
    expected = [{"Aisha Rahman", "Fatima Bello"}, {"Chloe Martin", "Daniel Kim"}, {"Nadia Haddad", "Ben Carter"},
                {"Priya Shah", "Tom Okafor"}]
    assert sum(frozenset(p) in pairs for p in expected) >= 3, pairs


def test_one_intro_per_member_and_consent_respected(demo):
    r = propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    names = _names(demo)
    counts = Counter(m for i in r.intros for m in (i.member_a_id, i.member_b_id))
    assert max(counts.values()) == 1
    matched = {names[m] for m in counts}
    assert "Olivia Brown" not in matched  # no intro consent
    assert "Henry Walsh" not in matched  # paused


def test_every_intro_cites_real_facts(demo):
    from kym.models import Fact

    r = propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    for intro in r.intros:
        assert intro.reasons, "an intro without reasons"
        for reason in intro.reasons:
            if reason["kind"] != "shared":
                assert reason["strength"] >= 0.3, f"weak reason shown: {reason}"
            for key in ("want", "offer", "a_fact", "b_fact"):
                if reason.get(key):
                    fact = demo.get(Fact, reason[key]["id"])
                    assert fact is not None and fact.text == reason[key]["text"]
            ids = {c_id for side in ("to_a", "to_b") for c in intro.copy[side] for c_id in c["reason_ids"]}
        assert ids <= {r_["id"] for r_ in intro.reasons}


def test_no_repeat_pairs_and_idempotent_week(demo):
    first = propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    again = propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    assert again.already and not again.intros
    from datetime import timedelta

    later = propose(demo, MONDAY + timedelta(days=7), use_llm=False, include_synthetic=True)
    before = {frozenset((i.member_a_id, i.member_b_id)) for i in first.intros}
    after = {frozenset((i.member_a_id, i.member_b_id)) for i in later.intros}
    assert not before & after


def test_synthetic_members_excluded_by_default(demo):
    r = propose(demo, MONDAY, use_llm=False)
    assert r.intros == [] and r.eligible == []


def test_people_who_already_talk_are_penalised(demo):
    from kym.graph import interaction_weights, pair

    names = {v: k for k, v in _names(demo).items()}
    weights = interaction_weights(demo)
    assert weights[pair(names["Lena Fischer"], names["Zara Ahmed"])] >= 5
    r = propose(demo, MONDAY, use_llm=False, include_synthetic=True)
    pairs = {frozenset((i.member_a_id, i.member_b_id)) for i in r.intros}
    assert frozenset((names["Lena Fischer"], names["Zara Ahmed"])) not in pairs
