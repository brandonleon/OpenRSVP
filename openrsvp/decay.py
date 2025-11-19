"""Decay cycle utilities."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from sqlalchemy import select

from .config import settings
from .database import engine, get_session
from .models import Channel, Event

SECONDS_PER_DAY = 60 * 60 * 24


def _elapsed_days(reference: datetime, now: datetime) -> float:
    delta = now - reference
    return max(delta.total_seconds() / SECONDS_PER_DAY, 0.0)


def run_decay_cycle() -> dict:
    """Apply exponential decay and purge expired entities."""
    stats = {
        "events_updated": 0,
        "events_deleted": 0,
        "channels_updated": 0,
        "channels_deleted": 0,
    }
    now = datetime.now(timezone.utc)
    with get_session() as session:
        events = session.scalars(select(Event)).all()
        for event in events:
            last_touch = event.last_accessed or event.created_at
            elapsed_days = _elapsed_days(last_touch, now)
            if elapsed_days <= 0:
                continue
            event.score = event.score * math.pow(settings.decay_factor, elapsed_days)
            event.last_modified = now
            session.add(event)
            stats["events_updated"] += 1
            for rsvp in list(event.rsvps):
                rsvp_elapsed = _elapsed_days(rsvp.last_modified or rsvp.created_at, now)
                if rsvp_elapsed > 0:
                    rsvp.score = rsvp.score * math.pow(
                        settings.decay_factor, rsvp_elapsed
                    )
                    rsvp.last_modified = now
                    session.add(rsvp)
            age_days = _elapsed_days(event.created_at, now)
            if (
                event.score <= settings.delete_threshold
                and age_days >= settings.delete_after_days
            ):
                session.delete(event)
                stats["events_deleted"] += 1

        channels = session.scalars(select(Channel)).all()
        for channel in channels:
            last_touch = channel.last_used_at or channel.created_at
            elapsed_days = _elapsed_days(last_touch, now)
            if elapsed_days <= 0:
                continue
            channel.score = channel.score * math.pow(
                settings.decay_factor, elapsed_days
            )
            session.add(channel)
            stats["channels_updated"] += 1
            if channel.score <= settings.delete_threshold and not channel.events:
                session.delete(channel)
                stats["channels_deleted"] += 1
        session.flush()
    return stats


def vacuum_database() -> None:
    with engine.connect() as connection:
        connection.execution_options(isolation_level="AUTOCOMMIT").exec_driver_sql(
            "VACUUM"
        )
