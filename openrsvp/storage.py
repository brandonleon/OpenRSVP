"""Database initialization and helpers."""

from __future__ import annotations

import secrets
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from .config import settings
from .database import engine, get_session
from .models import Base, Meta
from .utils import utcnow


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_schema_updates()
    ensure_root_token()


def ensure_schema_updates() -> None:
    """Perform lightweight schema updates for existing deployments."""
    with engine.begin() as conn:
        try:
            result = conn.exec_driver_sql("PRAGMA table_info(events)")
        except OperationalError:
            return
        columns = {row[1] for row in result}
        if "end_time" not in columns:
            conn.exec_driver_sql("ALTER TABLE events ADD COLUMN end_time DATETIME")


def ensure_root_token() -> str:
    with get_session() as session:
        existing = session.get(Meta, settings.root_token_key)
        if existing:
            return existing.value
        token = secrets.token_urlsafe(32)
        meta = Meta(key=settings.root_token_key, value=token, updated_at=utcnow())
        session.merge(meta)
        return token


def rotate_root_token() -> str:
    token = secrets.token_urlsafe(32)
    with get_session() as session:
        meta = Meta(key=settings.root_token_key, value=token, updated_at=utcnow())
        session.merge(meta)
    return token


def fetch_root_token() -> str:
    with get_session() as session:
        meta = session.get(Meta, settings.root_token_key)
        if not meta:
            return ensure_root_token()
        return meta.value
