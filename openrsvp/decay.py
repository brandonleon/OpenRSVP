"""Decay cycle utilities."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, or_, select

from .config import settings
from .database import engine, get_session
from .models import Channel, Event

# Use uvicorn's error logger so decay messages show up with level prefixes.
logger = logging.getLogger("uvicorn.error")

DECAY_BATCH_SIZE = 200

SECONDS_PER_DAY = 60 * 60 * 24


def _ensure_aware(dt: datetime) -> datetime:
    """Return a timezone aware dtetime (UTC) for arithmetic operations."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _elapsed_days(reference: datetime, now: datetime) -> float:
    delta = _ensure_aware(now) - _ensure_aware(reference)
    return max(delta.total_seconds() / SECONDS_PER_DAY, 0.0)


def _event_grace_expires_at(event: Event) -> datetime:
    """Return the datetime after which the event can be removed."""

    anchor = event.end_time or event.start_time or event.created_at
    return _ensure_aware(anchor) + timedelta(days=settings.delete_grace_days_after_end)


def _event_start_reference(event: Event) -> datetime:
    return _ensure_aware(event.start_time or event.created_at)


def _can_delete_event(event: Event, now: datetime, *, age_days: float) -> bool:
    """Check whether an event is eligible for deletion."""

    if settings.protect_upcoming_events and _event_start_reference(event) > _ensure_aware(
        now
    ):
        return False
    if _ensure_aware(now) < _event_grace_expires_at(event):
        return False
    if age_days < settings.delete_after_days:
        return False
    return event.score <= settings.delete_threshold


def run_decay_cycle() -> dict:
    """Apply exponential decay and purge expired entities."""
    stats = {
        "events_updated": 0,
        "events_deleted": 0,
        "channels_updated": 0,
        "channels_deleted": 0,
        "event_batches": 0,
        "channel_batches": 0,
        "events_skipped": 0,
        "channels_skipped": 0,
    }
    now = datetime.now(timezone.utc)
    decay_cutoff = now - timedelta(days=settings.delete_after_days)
    now_naive = now.replace(tzinfo=None)

    logger.info(
        "Decay cycle started (decay_factor=%.4f, delete_threshold=%.4f, delete_after_days=%d)",
        settings.decay_factor,
        settings.delete_threshold,
        settings.delete_after_days,
    )

    with get_session() as session:
        active_events_filter = or_(
            Event.score > settings.delete_threshold, Event.created_at >= decay_cutoff
        )
        delete_event_filter = and_(
            Event.score <= settings.delete_threshold, Event.created_at < decay_cutoff
        )
        if settings.protect_upcoming_events:
            delete_event_filter = and_(delete_event_filter, Event.start_time <= now_naive)
        covered_event_filter = or_(active_events_filter, delete_event_filter)
        total_events = session.scalar(select(func.count()).select_from(Event)) or 0
        covered_events = (
            session.scalar(
                select(func.count()).select_from(Event).where(covered_event_filter)
            )
            or 0
        )
        stats["events_skipped"] = max(total_events - covered_events, 0)

        last_event_seen: tuple[datetime | None, str | None] = (None, None)
        while True:
            query = (
                select(Event)
                .where(active_events_filter)
                .order_by(Event.created_at, Event.id)
            )
            if last_event_seen[0]:
                query = query.where(
                    or_(
                        Event.created_at > last_event_seen[0],
                        and_(
                            Event.created_at == last_event_seen[0],
                            Event.id > (last_event_seen[1] or ""),
                        ),
                    )
                )
            event_batch = session.scalars(query.limit(DECAY_BATCH_SIZE)).all()
            if not event_batch:
                break
            for event in event_batch:
                last_touch = event.last_accessed or event.created_at
                elapsed_days = _elapsed_days(last_touch, now)
                if elapsed_days <= 0:
                    continue
                previous_score = event.score
                event.score = event.score * math.pow(
                    settings.decay_factor, elapsed_days
                )
                event.last_modified = now
                session.add(event)
                logger.debug(
                    "Decayed event %s (%s): score %.4f -> %.4f after %.2f days",
                    event.id,
                    event.title,
                    previous_score,
                    event.score,
                    elapsed_days,
                )
                stats["events_updated"] += 1
                # RSVPs inherit lifecycle from their event; we avoid per-RSVP decay to reduce churn.
                age_days = _elapsed_days(event.created_at, now)
                if _can_delete_event(event, now, age_days=age_days):
                    logger.debug(
                        "Deleting event %s (%s): score %.4f age %.2f days",
                        event.id,
                        event.title,
                        event.score,
                        age_days,
                    )
                    session.delete(event)
                    stats["events_deleted"] += 1

            last_event_seen = (event_batch[-1].created_at, event_batch[-1].id)
            stats["event_batches"] += 1
            session.commit()

        last_event_seen = (None, None)
        while True:
            query = (
                select(Event)
                .where(delete_event_filter)
                .order_by(Event.created_at, Event.id)
            )
            if last_event_seen[0]:
                query = query.where(
                    or_(
                        Event.created_at > last_event_seen[0],
                        and_(
                            Event.created_at == last_event_seen[0],
                            Event.id > (last_event_seen[1] or ""),
                        ),
                    )
                )
            deletion_batch = session.scalars(query.limit(DECAY_BATCH_SIZE)).all()
            if not deletion_batch:
                break
            for event in deletion_batch:
                age_days = _elapsed_days(event.created_at, now)
                if _can_delete_event(event, now, age_days=age_days):
                    logger.debug(
                        "Deleting event %s (%s) below threshold with no prior decay pass",
                        event.id,
                        event.title,
                    )
                    session.delete(event)
                    stats["events_deleted"] += 1
            last_event_seen = (deletion_batch[-1].created_at, deletion_batch[-1].id)
            stats["event_batches"] += 1
            session.commit()

        active_channel_filter = or_(
            Channel.score > settings.delete_threshold,
            Channel.last_used_at >= decay_cutoff,
        )
        delete_channel_filter = Channel.score <= settings.delete_threshold
        covered_channel_filter = or_(active_channel_filter, delete_channel_filter)
        total_channels = session.scalar(select(func.count()).select_from(Channel)) or 0
        covered_channels = (
            session.scalar(
                select(func.count()).select_from(Channel).where(covered_channel_filter)
            )
            or 0
        )
        stats["channels_skipped"] = max(total_channels - covered_channels, 0)

        last_channel_seen: tuple[datetime | None, str | None] = (None, None)
        while True:
            query = (
                select(Channel)
                .where(active_channel_filter)
                .order_by(Channel.last_used_at, Channel.id)
            )
            if last_channel_seen[0]:
                query = query.where(
                    or_(
                        Channel.last_used_at > last_channel_seen[0],
                        and_(
                            Channel.last_used_at == last_channel_seen[0],
                            Channel.id > (last_channel_seen[1] or ""),
                        ),
                    )
                )
            channel_batch = session.scalars(query.limit(DECAY_BATCH_SIZE)).all()
            if not channel_batch:
                break
            for channel in channel_batch:
                last_touch = channel.last_used_at or channel.created_at
                elapsed_days = _elapsed_days(last_touch, now)
                if elapsed_days <= 0:
                    continue
                previous_score = channel.score
                channel.score = channel.score * math.pow(
                    settings.decay_factor, elapsed_days
                )
                session.add(channel)
                logger.debug(
                    "Decayed channel %s (%s): score %.4f -> %.4f after %.2f days",
                    channel.id,
                    channel.name,
                    previous_score,
                    channel.score,
                    elapsed_days,
                )
                stats["channels_updated"] += 1
                if channel.score <= settings.delete_threshold and not channel.events:
                    logger.debug(
                        "Deleting channel %s (%s): score %.4f (empty)",
                        channel.id,
                        channel.name,
                        channel.score,
                    )
                    session.delete(channel)
                    stats["channels_deleted"] += 1

            last_channel_seen = (channel_batch[-1].last_used_at, channel_batch[-1].id)
            stats["channel_batches"] += 1
            session.commit()

        last_channel_seen = (None, None)
        while True:
            query = (
                select(Channel)
                .where(delete_channel_filter)
                .order_by(Channel.last_used_at, Channel.id)
            )
            if last_channel_seen[0]:
                query = query.where(
                    or_(
                        Channel.last_used_at > last_channel_seen[0],
                        and_(
                            Channel.last_used_at == last_channel_seen[0],
                            Channel.id > (last_channel_seen[1] or ""),
                        ),
                    )
                )
            deletion_batch = session.scalars(query.limit(DECAY_BATCH_SIZE)).all()
            if not deletion_batch:
                break
            for channel in deletion_batch:
                if channel.score <= settings.delete_threshold and not channel.events:
                    logger.debug(
                        "Deleting channel %s (%s) below threshold (empty)",
                        channel.id,
                        channel.name,
                    )
                    session.delete(channel)
                    stats["channels_deleted"] += 1
            last_channel_seen = (
                deletion_batch[-1].last_used_at,
                deletion_batch[-1].id,
            )
            stats["channel_batches"] += 1
            session.commit()

        if stats["events_skipped"]:
            logger.warning(
                "Decay cycle skipped %d events outside filters; consider adjusting decay thresholds",
                stats["events_skipped"],
            )
        if stats["channels_skipped"]:
            logger.warning(
                "Decay cycle skipped %d channels outside filters; consider adjusting decay thresholds",
                stats["channels_skipped"],
            )

    logger.info(
        (
            "Decay cycle finished: events updated=%d, deleted=%d, skipped=%d across %d batches; "
            "channels updated=%d, deleted=%d, skipped=%d across %d batches"
        ),
        stats["events_updated"],
        stats["events_deleted"],
        stats["events_skipped"],
        stats["event_batches"],
        stats["channels_updated"],
        stats["channels_deleted"],
        stats["channels_skipped"],
        stats["channel_batches"],
    )
    return stats


def vacuum_database() -> None:
    with engine.connect() as connection:
        connection.execution_options(isolation_level="AUTOCOMMIT").exec_driver_sql(
            "VACUUM"
        )
