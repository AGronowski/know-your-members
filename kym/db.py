"""Database engine, sessions and a few column types.

SQLite (default, `data/kym.db`) for local runs; any Postgres URL (Supabase) via DATABASE_URL.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import DateTime, Engine, create_engine, event, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import TypeDecorator

from .config import get_settings


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend (SQLite stores them naive)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime passed to UTCDateTime column")
        value = value.astimezone(timezone.utc)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _normalise_url(url: str) -> str:
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def configure(url: str | None = None) -> Engine:
    """(Re)bind the module to a database. Tests and the simulator pass their own URL."""
    global _engine, _session_factory
    url = _normalise_url(url or get_settings().database_url)
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    else:
        # Supabase's pooler (pgbouncer) does not support prepared statements.
        kwargs["connect_args"] = {"prepare_threshold": None}
        kwargs["pool_pre_ping"] = True
    if _engine is not None:
        _engine.dispose()
    _engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    _session_factory = sessionmaker(_engine, expire_on_commit=False)
    return _engine


def engine() -> Engine:
    if _engine is None:
        configure()
    return _engine


def init_db() -> None:
    from . import models  # noqa: F401  (registers tables)

    Base.metadata.create_all(engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    if _session_factory is None:
        configure()
    session = _session_factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def get_setting(session: Session, key: str, default: str | None = None) -> str | None:
    from .models import Setting

    row = session.get(Setting, key)
    return row.value if row else default


def set_setting(session: Session, key: str, value: str | None) -> None:
    from .models import Setting

    row = session.get(Setting, key)
    if value is None:
        if row:
            session.delete(row)
        return
    if row:
        row.value = value
    else:
        session.add(Setting(key=key, value=value))


def settings_with_prefix(session: Session, prefix: str) -> dict[str, str]:
    from .models import Setting

    rows = session.scalars(select(Setting).where(Setting.key.startswith(prefix)))
    return {r.key: r.value for r in rows}
