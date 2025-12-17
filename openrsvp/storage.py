"""Database initialization and helpers."""

from __future__ import annotations

import secrets
import shutil
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from .config import settings
from .database import engine, get_session
from .models import Meta
from .utils import utcnow


def init_db() -> None:
    upgrade_database(make_backup=False)
    ensure_root_token()


def ensure_schema_updates() -> list[str]:
    """Perform lightweight schema updates for existing SQLite deployments."""
    actions: list[str] = []
    if engine.dialect.name != "sqlite":
        return actions

    inspector = inspect(engine)
    if not inspector.has_table("events"):
        return actions

    columns = {col["name"] for col in inspector.get_columns("events")}
    if "end_time" not in columns:
        with engine.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE events ADD COLUMN end_time DATETIME")
            actions.append("Added events.end_time column")
    return actions


def _alembic_config() -> Config:
    package_dir = Path(__file__).resolve().parent
    script_location = package_dir / "alembic"
    ini_path = script_location.parent / "alembic.ini"

    config = Config(str(ini_path)) if ini_path.exists() else Config()
    config.set_main_option("script_location", str(script_location))
    config.set_main_option("sqlalchemy.url", str(engine.url))
    return config


def upgrade_database(*, make_backup: bool = True) -> list[str]:
    """Upgrade the database schema in-place.

    Returns a list of applied actions; empty if already up-to-date.
    """
    actions: list[str] = []
    db_path = Path(settings.database_path)

    if make_backup and db_path.exists():
        backup_path = db_path.with_suffix(db_path.suffix + ".bak")
        shutil.copy(db_path, backup_path)
        actions.append(f"Backup created at {backup_path}")

    inspector = inspect(engine)
    has_alembic = inspector.has_table("alembic_version")
    has_events = inspector.has_table("events")
    config = _alembic_config()

    # Run minimal legacy fixes for pre-Alembic SQLite databases.
    actions.extend(ensure_schema_updates())

    if not has_alembic and not has_events:
        # Fresh database: run migrations normally.
        command.upgrade(config, "head")
        actions.append("Ran Alembic upgrade to head (fresh database)")
    elif not has_alembic:
        # Existing database without Alembic tracking: baseline it.
        command.stamp(config, "head")
        actions.append("Stamped existing database to Alembic head")
    else:
        command.upgrade(config, "head")
        actions.append("Applied Alembic migrations to head")

    return actions


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
