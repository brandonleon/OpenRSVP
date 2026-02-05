"""Shared pytest fixtures for OpenRSVP."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openrsvp import api, database, storage
from openrsvp.models import Base


@pytest.fixture(scope="session", autouse=True)
def configure_in_memory_db():
    """Reuse a single in-memory SQLite database for fast, isolated tests."""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    session_factory = scoped_session(
        sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            future=True,
            expire_on_commit=False,
        )
    )
    database.engine = engine
    database.SessionLocal = session_factory
    storage.engine = engine
    storage.get_session = database.get_session
    api.SessionLocal = session_factory
    Base.metadata.create_all(bind=engine)
    yield
    session_factory.remove()


@pytest.fixture(autouse=True)
def clean_database():
    """Reset all tables between tests to guarantee isolation."""

    Base.metadata.drop_all(bind=database.engine)
    Base.metadata.create_all(bind=database.engine)
    yield
