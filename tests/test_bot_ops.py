"""The bot's database layer, exercised without Discord: the same calls the bot makes, in the same order."""

from datetime import timedelta

from sqlalchemy import select, update

from kym import db
from kym.bot import ops
from kym.models import Member, Message

from .conftest import MONDAY


def _make_real(session, names):
    """Pretend two synthetic members are real Discord users so the live cycle (which skips synthetic ones) uses them."""
    session.execute(update(Member).where(Member.name.in_(names)).values(source="discord"))
    session.execute(update(Member).where(Member.name.not_in(names)).values(status="left"))


def test_full_intro_round_trip_through_ops(demo):
    _make_real(demo, ["Chloe Martin", "Daniel Kim"])
    demo.commit()
    r = ops.run_propose(MONDAY)
    assert r["intros"] == 1 and not r["already"]
    proposed = ops.proposed_intros(r["week"])
    payload = ops.intro_payload(proposed[0]["id"])
    assert payload["sides"]["a"]["lines"] and payload["sides"]["b"]["lines"]
    intro_id = payload["id"]
    assert intro_id in ops.due_offers(MONDAY + timedelta(days=1))
    ops.mark_offered(intro_id, MONDAY + timedelta(days=1), {"a": {"via": "dm"}, "b": {"via": "dm"}})
    a_id, b_id = payload["sides"]["a"]["discord_id"], payload["sides"]["b"]["discord_id"]
    assert ops.respond(intro_id, 12345, "accept", MONDAY)["outcome"] == "not_yours"
    assert ops.respond(intro_id, a_id, "accept", MONDAY + timedelta(days=1))["outcome"] == "waiting"
    assert ops.respond(intro_id, b_id, "accept", MONDAY + timedelta(days=1))["outcome"] == "connect"
    ops.mark_connected(intro_id, 777, MONDAY + timedelta(days=1))
    due = ops.due_followups(MONDAY + timedelta(days=6))
    assert [d["id"] for d in due] == [intro_id] and due[0]["thread_id"] == 777
    ops.mark_followed_up(intro_id, MONDAY + timedelta(days=6))
    assert not ops.feedback(intro_id, 12345, "useful", MONDAY)
    assert ops.feedback(intro_id, b_id, "useful", MONDAY + timedelta(days=6))
    assert [d["id"] for d in ops.due_outcomes(MONDAY + timedelta(days=9))] == [intro_id]
    assert ops.finalize(intro_id, MONDAY + timedelta(days=9), 5, 4) == "useful"
    digest = ops.make_digest(r["week"], publish=False)
    assert any("Intros: 1 proposed" in line for line in digest["lines"])


def test_history_mirror_accepts_discord_style_rows(demo):
    row = {"id": 1, "guild_id": 9, "channel_id": 10, "channel_name": "help", "thread_id": None, "thread_name": None,
           "author_id": 100000000000000001, "author_name": "Priya", "author_is_bot": False, "content": "hello?",
           "created_at": MONDAY, "reply_to_id": None, "mention_ids": [], "jump_url": "https://discord.com/x"}
    demo.commit()
    assert ops.store_messages([row], {10: 1}) == 1
    assert ops.store_messages([row], {}) == 1  # duplicates are ignored, not errors
    with db.session_scope() as s:
        msg = s.scalar(select(Message).where(Message.id == 1))
        assert msg.created_at == MONDAY
    assert ops.last_synced(10) == 1


def test_consent_and_forget(demo):
    demo.commit()
    did = 555
    result = ops.do_onboard(did, "New Person", {"building": "A voice agent", "offers": "TTS tuning", "wants": "first users",
                                                 "speak": "", "links": "github.com/newperson"}, {"store", "intro"})
    assert result["handles"] == {"github": "newperson"} and not result["enrich"]
    assert ops.my_profile(did)["consent"]["intro"] is True
    assert ops.change_frequency(did, "paused")
    assert ops.forget_me(did)
    profile = ops.my_profile(did)
    assert profile["facts"] == [] and not any(profile["consent"].values()) and profile["status"] == "left"
