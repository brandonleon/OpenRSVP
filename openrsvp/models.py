"""SQLAlchemy models for OpenRSVP."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

from .utils import utcnow

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return utcnow()


class Meta(Base):
    __tablename__ = "meta"

    key = Column(String(128), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=_now, onupdate=_now, nullable=False)


class Event(Base):
    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=_uuid)
    admin_token = Column(String(128), nullable=False, unique=True)
    channel_id = Column(String(36), ForeignKey("channels.id"), nullable=True)
    is_private = Column(Boolean, default=False, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    location = Column(String(255), nullable=True)
    score = Column(Float, default=100.0, nullable=False)
    created_at = Column(DateTime, default=_now, nullable=False)
    last_modified = Column(DateTime, default=_now, onupdate=_now, nullable=False)
    last_accessed = Column(DateTime, default=_now, onupdate=_now, nullable=False)

    channel = relationship("Channel", back_populates="events")
    rsvps = relationship("RSVP", back_populates="event", cascade="all, delete-orphan")


class RSVP(Base):
    __tablename__ = "rsvps"

    id = Column(String(36), primary_key=True, default=_uuid)
    event_id = Column(String(36), ForeignKey("events.id"), nullable=False)
    rsvp_token = Column(String(128), nullable=False, unique=True)
    name = Column(String(120), nullable=False)
    pronouns = Column(String(64))
    status = Column(String(16), nullable=False, default="yes")
    guest_count = Column(Integer, default=0, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=_now, nullable=False)
    last_modified = Column(DateTime, default=_now, onupdate=_now, nullable=False)

    event = relationship("Event", back_populates="rsvps")


class Channel(Base):
    __tablename__ = "channels"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=False)
    slug = Column(String(128), nullable=False, unique=True)
    visibility = Column(String(16), nullable=False, default="public")
    score = Column(Float, default=100.0, nullable=False)
    created_at = Column(DateTime, default=_now, nullable=False)
    last_used_at = Column(DateTime, default=_now, nullable=False)

    events = relationship("Event", back_populates="channel", cascade="all, delete")
