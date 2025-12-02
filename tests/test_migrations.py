from __future__ import annotations

import types

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from openrsvp import storage, database
from openrsvp.models import Base


def _patch_db(monkeypatch: pytest.MonkeyPatch, engine: Engine, db_path) -> None:
    monkeypatch.setattr(storage, "engine", engine)
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "DATABASE_URL", str(engine.url))
    fake_settings = types.SimpleNamespace(database_path=db_path)
    monkeypatch.setattr(storage, "settings", fake_settings)


def _get_version(engine: Engine) -> str | None:
    with engine.connect() as conn:
        try:
            return conn.execute(
                text("select version_num from alembic_version")
            ).scalar()
        except Exception:
            return None


def test_upgrade_database_stamps_existing_db(monkeypatch, tmp_path):
    db_path = tmp_path / "existing.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(bind=engine)  # existing schema without Alembic tracking
    _patch_db(monkeypatch, engine, db_path)

    actions = storage.upgrade_database(make_backup=False)

    assert "Stamped existing database to Alembic head" in actions
    assert _get_version(engine) == "0001_initial"


def test_upgrade_database_creates_fresh_schema(monkeypatch, tmp_path):
    db_path = tmp_path / "fresh.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    _patch_db(monkeypatch, engine, db_path)

    actions = storage.upgrade_database(make_backup=False)

    assert "Ran Alembic upgrade to head (fresh database)" in actions
    assert _get_version(engine) == "0001_initial"
    inspector = inspect(engine)
    assert inspector.has_table("events")
    assert inspector.has_table("channels")
    assert inspector.has_table("rsvps")
    assert inspector.has_table("messages")
