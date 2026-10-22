"""Onboarding, consent and data deletion."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .db import utcnow
from .facts import Extraction, extract_answers, replace_facts
from .models import CONSENT_SCOPES, Answer, ConsentEvent, Fact, Interaction, Member

REQUIRED_SCOPES = {"store", "intro"}


@dataclass(frozen=True)
class Question:
    key: str
    label: str  # Discord modal labels are capped at 45 characters
    placeholder: str
    paragraph: bool
    required: bool
    max_length: int


QUESTIONS = [
    Question("building", "What are you building right now?",
             "e.g. an open-source eval harness for RAG apps", True, True, 600),
    Question("offers", "What could you help someone with in 30 min?",
             "e.g. React performance, pricing a B2B product, first UK hires", True, True, 500),
    Question("wants", "What do you want help with / who to meet?",
             "e.g. a designer to pair with, intros to climate angels", True, True, 500),
    Question("speak", "Would you speak or demo? On what?",
             "e.g. yes: lessons from shipping voice agents (blank = no)", False, False, 200),
    Question("links", "GitHub / Devpost / HN / site (optional)",
             "e.g. github.com/you devpost.com/you", False, False, 300),
]


def set_consent(session: Session, member: Member, scopes: dict[str, bool], source: str) -> None:
    current = dict(member.consent or {})
    for scope, granted in scopes.items():
        if scope not in CONSENT_SCOPES:
            raise ValueError(f"unknown consent scope {scope!r}")
        if current.get(scope) != granted:
            session.add(ConsentEvent(member_id=member.id, scope=scope, granted=granted, at=utcnow(), source=source))
        current[scope] = granted
    member.consent = current


def get_or_create_discord_member(session: Session, discord_id: int, name: str) -> Member:
    member = session.scalar(select(Member).where(Member.discord_id == discord_id))
    if member is None:
        member = Member(discord_id=discord_id, name=name, source="discord", status="active", handles={}, consent={}, meta={})
        session.add(member)
        session.flush()
    else:
        member.name = name
        if member.status in ("left", "prospect"):
            member.status = "active"
    return member


def onboard(session: Session, member: Member, answers: dict[str, str], scopes: set[str], source: str,
            use_llm: bool = True) -> Extraction:
    """Store answers + consent, extract facts. Requires the store and intro scopes."""
    missing = REQUIRED_SCOPES - scopes
    if missing:
        raise ValueError(f"missing required consent: {', '.join(sorted(missing))}")
    set_consent(session, member, {scope: scope in scopes for scope in CONSENT_SCOPES}, source)
    session.execute(delete(Answer).where(Answer.member_id == member.id))
    for key, text in answers.items():
        if text and text.strip():
            session.add(Answer(member_id=member.id, question=key, text=text.strip()))
    extraction = extract_answers(session, answers, use_llm=use_llm)
    replace_facts(session, member, extraction.facts, {"answer"})
    member.handles = {**(member.handles or {}), **extraction.handles}
    member.onboarded_at = utcnow()
    member.status = "active"
    member.meta = {**(member.meta or {}), "would_speak": extraction.would_speak, "extractor": extraction.extractor}
    return extraction


def set_frequency(member: Member, frequency: str) -> None:
    if frequency not in ("weekly", "fortnightly", "paused"):
        raise ValueError(frequency)
    member.intro_frequency = frequency


def forget(session: Session, member: Member, source: str) -> None:
    """Delete everything we hold about a member except the consent audit trail and a tombstone row."""
    set_consent(session, member, {scope: False for scope in CONSENT_SCOPES}, source)
    session.execute(delete(Fact).where(Fact.member_id == member.id))
    session.execute(delete(Answer).where(Answer.member_id == member.id))
    session.execute(delete(Interaction).where((Interaction.member_a_id == member.id) | (Interaction.member_b_id == member.id)))
    member.handles = {}
    member.email = None
    member.meta = {"forgotten_at": utcnow().isoformat()}
    member.status = "left"
    member.intro_frequency = "paused"


def profile_summary(session: Session, member: Member) -> dict:
    facts = session.scalars(select(Fact).where(Fact.member_id == member.id).order_by(Fact.kind, Fact.id)).all()
    return {
        "name": member.name,
        "status": member.status,
        "frequency": member.intro_frequency,
        "consent": {k: bool((member.consent or {}).get(k)) for k in CONSENT_SCOPES},
        "handles": member.handles or {},
        "facts": [{"kind": f.kind, "text": f.text, "source": f.source, "url": f.url, "tags": (f.meta or {}).get("tags", [])}
                  for f in facts],
    }
