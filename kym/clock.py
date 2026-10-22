"""Weeks and schedule points in the community's local timezone."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from .config import get_settings


def _tz(tz: str | None) -> ZoneInfo:
    return ZoneInfo(tz or get_settings().timezone)


def week_id(dt: datetime, tz: str | None = None) -> str:
    year, week, _ = dt.astimezone(_tz(tz)).isocalendar()
    return f"{year}-W{week:02d}"


def week_start(week: str, tz: str | None = None) -> datetime:
    """Monday 00:00 local time of an ISO week, returned in UTC."""
    year, w = week.split("-W")
    local = datetime.fromisocalendar(int(year), int(w), 1).replace(tzinfo=_tz(tz))
    return local.astimezone(timezone.utc)


def at(week: str, weekday: int, hour: int, tz: str | None = None) -> datetime:
    """A point in an ISO week (weekday 0 = Monday), local wall-clock time, returned in UTC."""
    year, w = week.split("-W")
    local = datetime.fromisocalendar(int(year), int(w), weekday + 1).replace(hour=hour, tzinfo=_tz(tz))
    return local.astimezone(timezone.utc)


def previous_week(week: str, tz: str | None = None) -> str:
    return week_id(week_start(week, tz) - timedelta(days=3), tz)


def next_week(week: str, tz: str | None = None) -> str:
    return week_id(week_start(week, tz) + timedelta(days=10), tz)
