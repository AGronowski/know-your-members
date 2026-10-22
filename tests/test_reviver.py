from datetime import timedelta

from sqlalchemy import select

from kym import reviver
from kym.models import Message, Revival

from .conftest import MONDAY


def test_detects_open_questions_not_small_talk_or_resolved(demo):
    titles = {c.conv.title: c.rule for c in reviver.detect(demo, MONDAY)}
    assert titles.keys() >= {"LiveKit vs Pipecat for low-latency voice?", "Postgres or ClickHouse for product analytics events?",
                             "ragcheck v0.2: eval harness for RAG", "Co-organiser for an AI agents meetup?",
                             "R&D tax credits for LLM work?"}
    assert "Happy Friday" not in titles  # small talk
    assert "GitHub student pack?" not in titles  # resolved
    assert "What's everyone shipping this week?" not in titles  # still active


def test_scan_logs_every_candidate_and_only_tags_consenting_members(demo):
    rows = reviver.scan(demo, MONDAY)
    assert rows and all(r.status in ("drafted", "skipped") for r in rows)
    for r in rows:
        text, mentions = reviver.render(r) if r.draft else ("", [])
        for h in r.judge.get("helpers", []):
            assert h["discord_id"] in mentions
        assert "@everyone" not in text and "@here" not in text
    postgres = next(r for r in rows if r.title.startswith("Postgres"))
    assert all(h["name"] == "Emily Zhang" for h in postgres.judge["helpers"])


def test_best_helper_is_the_member_whose_offer_fits_the_open_question(demo):
    cand = next(c for c in reviver.detect(demo, MONDAY) if c.conv.title.startswith("Postgres"))
    helpers = reviver._helpers(demo, cand.conv, cand.anchor)
    assert helpers and helpers[0]["name"] == "Emily Zhang"
    assert "Hannah Lee" not in {h["name"] for h in helpers} and "Tom Okafor" not in {h["name"] for h in helpers}


def test_not_rescanned_within_30_days(demo):
    first = reviver.scan(demo, MONDAY)
    assert first
    assert reviver.scan(demo, MONDAY + timedelta(days=1)) == []


def test_outcome_counts_human_replies(demo):
    rev = next(r for r in reviver.scan(demo, MONDAY) if r.status == "drafted")
    reviver.mark_posted(demo, rev, 999, MONDAY)
    assert reviver.evaluate(demo, MONDAY + timedelta(hours=2)) == []
    demo.add(Message(id=5, channel_id=rev.channel_id, thread_id=rev.thread_id, author_id=42, author_name="someone",
                     author_is_bot=False, content="Try ClickHouse with materialised views", created_at=MONDAY + timedelta(hours=3),
                     mention_ids=[]))
    demo.flush()
    assert reviver.evaluate(demo, MONDAY + timedelta(hours=4)) == [rev]
    assert rev.status == "replied" and rev.replies_48h == 1


def test_nudge_that_gets_no_reply_is_logged_as_a_failure(demo):
    rev = next(r for r in reviver.scan(demo, MONDAY) if r.status == "drafted")
    reviver.mark_posted(demo, rev, 999, MONDAY)
    reviver.evaluate(demo, MONDAY + timedelta(days=8))
    assert rev.status == "no_reply" and "nobody replied" in rev.failure_reason
    assert demo.scalar(select(Revival).where(Revival.status == "no_reply"))
