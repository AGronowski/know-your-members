"""The bot log. Every action (and every failure) gets a row the lead can read."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from .db import utcnow
from .models import Attempt

logger = logging.getLogger("kym")


def record(session: Session, kind: str, status: str, detail: str = "", target: str = "", **data) -> Attempt:
    attempt = Attempt(at=data.pop("at", None) or utcnow(), kind=kind, status=status, detail=detail[:4000],
                      target=target[:200], data=data)
    session.add(attempt)
    level = logging.WARNING if status == "failed" else logging.INFO
    logger.log(level, "%s %s %s %s", kind, status, target, detail[:200])
    return attempt
