"""Tables. Every fact about a person carries its source, so every claim the bot makes can link to it."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, UTCDateTime, utcnow

# Consent scopes a member can grant. `store` and `intro` are required to take part at all.
CONSENT_SCOPES = {
    "store": "Store my answers so the bot can use them",
    "intro": "Suggest one introduction a week to me",
    "enrich": "Read my public GitHub / Devpost / Hacker News profiles",
    "speaker": "Suggest me to organisers as a speaker or demoer",
    "tag": "Tag me in old threads where I could help",
}

FACT_KINDS = {"building", "offers", "wants", "skill", "domain", "shipped", "hosted", "spoke", "speak_topic", "location"}


class Member(Base):
    __tablename__ = "members"

    id: Mapped[int] = mapped_column(primary_key=True)
    discord_id: Mapped[int | None] = mapped_column(BigInteger, unique=True)
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(40), default="discord")  # discord | import | synthetic | prospect
    status: Mapped[str] = mapped_column(String(20), default="active")  # prospect | active | left
    email: Mapped[str | None] = mapped_column(String(320))
    handles: Mapped[dict] = mapped_column(JSON, default=dict)  # github, devpost, hn, website, linkedin, twitter
    consent: Mapped[dict] = mapped_column(JSON, default=dict)  # scope -> bool (current state; history in consent_events)
    intro_frequency: Mapped[str] = mapped_column(String(20), default="weekly")  # weekly | fortnightly | paused
    responsiveness: Mapped[float] = mapped_column(Float, default=0.7)
    joined_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    onboarded_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    notion_page_id: Mapped[str | None] = mapped_column(String(64))
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    facts: Mapped[list[Fact]] = relationship(back_populates="member", cascade="all, delete-orphan")

    def allows(self, scope: str) -> bool:
        return self.status == "active" and bool((self.consent or {}).get(scope))


class ConsentEvent(Base):
    """Append-only audit trail of every consent change."""

    __tablename__ = "consent_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    member_id: Mapped[int] = mapped_column(ForeignKey("members.id", ondelete="CASCADE"), index=True)
    scope: Mapped[str] = mapped_column(String(20))
    granted: Mapped[bool] = mapped_column(Boolean)
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    source: Mapped[str] = mapped_column(String(80))


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    member_id: Mapped[int] = mapped_column(ForeignKey("members.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(String(40))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Fact(Base):
    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(primary_key=True)
    member_id: Mapped[int] = mapped_column(ForeignKey("members.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(500))
    source: Mapped[str] = mapped_column(String(20))  # answer | github | devpost | hn | luma | import
    source_ref: Mapped[str | None] = mapped_column(String(200))
    observed_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    member: Mapped[Member] = relationship(back_populates="facts")


class Interaction(Base):
    """Who already talks to whom (from chat history), so intros go to people who don't know each other yet."""

    __tablename__ = "interactions"
    __table_args__ = (UniqueConstraint("member_a_id", "member_b_id", "source"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    member_a_id: Mapped[int] = mapped_column(ForeignKey("members.id", ondelete="CASCADE"), index=True)
    member_b_id: Mapped[int] = mapped_column(ForeignKey("members.id", ondelete="CASCADE"), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    last_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    source: Mapped[str] = mapped_column(String(20), default="chat")


class Intro(Base):
    __tablename__ = "intros"
    __table_args__ = (UniqueConstraint("member_a_id", "member_b_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    week: Mapped[str] = mapped_column(String(10), index=True)
    member_a_id: Mapped[int] = mapped_column(ForeignKey("members.id", ondelete="CASCADE"), index=True)
    member_b_id: Mapped[int] = mapped_column(ForeignKey("members.id", ondelete="CASCADE"), index=True)
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(20))
    breakdown: Mapped[dict] = mapped_column(JSON, default=dict)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    copy: Mapped[dict] = mapped_column(JSON, default=dict)  # to_a, to_b, opener, writer, dropped
    # proposed -> vetoed | offered -> declined | expired | connected -> completed ; failed on delivery errors;
    # preview: demo intro between two synthetic members, shown to the lead and never sent
    status: Mapped[str] = mapped_column(String(20), default="proposed", index=True)
    a_response: Mapped[str | None] = mapped_column(String(10))
    b_response: Mapped[str | None] = mapped_column(String(10))
    a_responded_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    b_responded_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    offered_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    connected_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    thread_id: Mapped[int | None] = mapped_column(BigInteger)
    followed_up_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    feedback: Mapped[dict] = mapped_column(JSON, default=dict)  # member_id -> useful | meh | didnt_happen
    messages_a: Mapped[int | None] = mapped_column(Integer)
    messages_b: Mapped[int | None] = mapped_column(Integer)
    outcome: Mapped[str | None] = mapped_column(String(20))
    vetoed: Mapped[bool] = mapped_column(Boolean, default=False)
    failure: Mapped[str | None] = mapped_column(Text)
    notion_page_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    member_a: Mapped[Member] = relationship(foreign_keys=[member_a_id])
    member_b: Mapped[Member] = relationship(foreign_keys=[member_b_id])

    def side(self, member_id: int) -> str:
        if member_id == self.member_a_id:
            return "a"
        if member_id == self.member_b_id:
            return "b"
        raise ValueError(f"member {member_id} is not part of intro {self.id}")


class Message(Base):
    """Mirror of Discord history (or an imported export) used by the thread reviver and the interaction graph."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    guild_id: Mapped[int | None] = mapped_column(BigInteger)
    channel_id: Mapped[int] = mapped_column(BigInteger, index=True)
    channel_name: Mapped[str | None] = mapped_column(String(200))
    thread_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    thread_name: Mapped[str | None] = mapped_column(String(200))
    author_id: Mapped[int] = mapped_column(BigInteger, index=True)
    author_name: Mapped[str] = mapped_column(String(200))
    author_is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    content: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    reply_to_id: Mapped[int | None] = mapped_column(BigInteger)
    mention_ids: Mapped[list] = mapped_column(JSON, default=list)
    jump_url: Mapped[str | None] = mapped_column(String(300))


class Revival(Base):
    __tablename__ = "revivals"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_key: Mapped[str] = mapped_column(String(80), index=True)
    channel_id: Mapped[int] = mapped_column(BigInteger)
    thread_id: Mapped[int | None] = mapped_column(BigInteger)
    anchor_message_id: Mapped[int | None] = mapped_column(BigInteger)
    rule: Mapped[str] = mapped_column(String(30))
    title: Mapped[str] = mapped_column(Text)
    excerpt: Mapped[str] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    judge: Mapped[dict] = mapped_column(JSON, default=dict)
    draft: Mapped[str | None] = mapped_column(Text)
    writer: Mapped[str | None] = mapped_column(String(20))
    tagged_member_ids: Mapped[list] = mapped_column(JSON, default=list)
    # candidate -> skipped | drafted -> posted -> replied | no_reply ; failed on errors
    status: Mapped[str] = mapped_column(String(20), default="candidate", index=True)
    posted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    nudge_message_id: Mapped[int | None] = mapped_column(BigInteger)
    replies_48h: Mapped[int | None] = mapped_column(Integer)
    replies_7d: Mapped[int | None] = mapped_column(Integer)
    distinct_repliers: Mapped[int | None] = mapped_column(Integer)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    evaluated_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    notion_page_id: Mapped[str | None] = mapped_column(String(64))


class Event(Base):
    """A Luma event with its hosts and featured guests (speaker evidence)."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # Luma api_id
    name: Mapped[str] = mapped_column(Text)
    start_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    url: Mapped[str | None] = mapped_column(String(300))
    city: Mapped[str | None] = mapped_column(String(100))
    calendar_name: Mapped[str | None] = mapped_column(String(200))
    calendar_api_id: Mapped[str | None] = mapped_column(String(40))
    guest_count: Mapped[int | None] = mapped_column(Integer)
    hosts: Mapped[list] = mapped_column(JSON, default=list)
    featured: Mapped[list] = mapped_column(JSON, default=list)
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(80))
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Attempt(Base):
    """The bot log: everything it tried, and how it went. Failures are first-class."""

    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    target: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(12))  # ok | failed | skipped
    detail: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    notion_page_id: Mapped[str | None] = mapped_column(String(64))


class SpeakerSearch(Base):
    __tablename__ = "speaker_searches"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expansion: Mapped[dict] = mapped_column(JSON, default=dict)
    results: Mapped[list] = mapped_column(JSON, default=list)
    notes: Mapped[list] = mapped_column(JSON, default=list)
    notion_page_id: Mapped[str | None] = mapped_column(String(64))


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class HttpCache(Base):
    __tablename__ = "http_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    url: Mapped[str] = mapped_column(Text)
    status: Mapped[int] = mapped_column(Integer)
    body: Mapped[str] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary)


class LLMCache(Base):
    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(20))
    response: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
