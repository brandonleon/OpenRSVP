"""CRUD helpers for events, RSVPs, and channels."""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session
import secrets

from .config import settings
from .models import Channel, Event, Message, RSVP
from .utils import slugify, to_naive_utc, utcnow

CHANNEL_VISIBILITIES = {"public", "private"}
VALID_APPROVAL_STATUSES = {"pending", "approved", "rejected"}
VALID_ATTENDANCE_STATUSES = {"yes", "no", "maybe"}


def _now() -> datetime:
    return utcnow()


def get_channel_by_slug(session: Session, slug: str) -> Channel | None:
    normalized = (slug or "").strip().lower()
    if not normalized:
        return None
    stmt = select(Channel).where(Channel.slug == normalized)
    return session.scalars(stmt).first()


def get_public_channels(
    session: Session, search_term: str | None = None, limit: int | None = None
) -> Sequence[Channel]:
    stmt = (
        select(Channel)
        .where(Channel.visibility == "public")
        .order_by(Channel.score.desc(), Channel.name.asc())
    )

    if search_term:
        stmt = stmt.where(
            Channel.name.ilike(f"%{search_term}%")
            | Channel.slug.ilike(f"%{search_term}%")
        )

    if limit and limit > 0:
        stmt = stmt.limit(limit)

    return session.scalars(stmt).all()


def touch_channel(channel: Channel, *, timestamp: datetime | None = None) -> None:
    channel.last_used_at = timestamp or _now()


def ensure_channel(
    session: Session,
    *,
    name: str,
    visibility: str,
) -> Channel:
    """Return an existing channel or create a new one."""
    visibility = visibility.lower()
    if visibility not in CHANNEL_VISIBILITIES:
        raise ValueError("Invalid channel visibility")
    slug = slugify(name)
    if not slug:
        raise ValueError("Invalid channel name")
    channel = get_channel_by_slug(session, slug)
    if channel:
        if channel.visibility != visibility:
            raise ValueError("Channel already exists with different visibility")
        channel.name = name
        touch_channel(channel)
        session.add(channel)
        session.flush()
        return channel
    channel = Channel(
        name=name,
        slug=slug,
        visibility=visibility,
        score=settings.initial_channel_score,
        created_at=_now(),
        last_used_at=_now(),
    )
    session.add(channel)
    session.flush()
    return channel


def get_admin_events_list(
    session: Session, channel_id: int | None = None, limit: int | None = None
) -> Sequence[Event]:
    """Retrieve events for admin management."""
    stmt = select(Event).order_by(Event.start_time.desc())

    if channel_id is not None:
        stmt = stmt.where(Event.channel_id == channel_id)

    if limit and limit > 0:
        stmt = stmt.limit(limit)

    return session.scalars(stmt).all()


def get_admin_approval_counts(session: Session, event_id: int) -> dict:
    """Get approval status counts for event."""
    # Get counts for each approval status
    approval_counts = (
        session.query(RSVP.approval_status, func.count())
        .filter(RSVP.event_id == event_id)
        .group_by(RSVP.approval_status)
        .all()
    )

    # Convert to dictionary
    counts = {status: 0 for status in VALID_APPROVAL_STATUSES}

    for status, count in approval_counts:
        counts[status] = count

    return counts


def validate_admin_token(
    session: Session, admin_token: str, model_type: type = Event
) -> bool:
    """Validate admin token for a given model."""
    result = (
        session.query(model_type).filter(model_type.admin_token == admin_token).first()
    )
    return result is not None

def create_event(
    session: Session,
    *,
    title: str,
    description: str | None,
    start_time: datetime,
    end_time: datetime | None,
    location: str | None = None,
    channel: Channel | None = None,
    admin_approval_required: bool = False,
    is_private: bool = False,
    max_attendees: int | None = None,
    rsvps_closed: bool = False,
    rsvp_close_at: datetime | None = None
) -> Event:
    """Create and persist a new event."""
    normalized_start = to_naive_utc(start_time)
    normalized_end = to_naive_utc(end_time)
    normalized_close = to_naive_utc(rsvp_close_at)
    
    # Generate a slug from title
    slug = slugify(title)

    event = Event(
        admin_token=secrets.token_urlsafe(32),
        is_private=is_private,
        admin_approval_required=admin_approval_required,
        rsvps_closed=rsvps_closed,
        rsvp_close_at=normalized_close,
        max_attendees=max_attendees,
        title=title,
        description=description,
        slug=slug,
        start_time=normalized_start,
        end_time=normalized_end,
        location=location,
        score=settings.initial_event_score,
        channel=channel,
    )
    session.add(event)
    session.flush()
    return event


def create_rsvp(
    session: Session,
    *,
    event: Event,
    name: str,
    attendance_status: str,
    pronouns: str | None = None,
    guest_count: int | None = None,
    is_private: bool = False,
    approval_status: str | None = None
) -> RSVP:
    """Create a new RSVP, respecting approval rules."""
    normalized_attendance = _normalize_attendance(attendance_status)
    normalized_approval = (
        _normalize_approval(approval_status)
        if approval_status is not None
        else ("pending" if event.admin_approval_required else "approved")
    )
    rsvp = RSVP(
        event=event,
        rsvp_token=secrets.token_urlsafe(32),
        name=name,
        attendance_status=normalized_attendance,
        approval_status=normalized_approval,
        pronouns=pronouns,
        guest_count=guest_count or 0,
        is_private=is_private,
    )
    session.add(rsvp)
    session.flush()
    return rsvp


def create_message(
    session: Session,
    *,
    event: Event | None,
    rsvp: RSVP | None,
    author_id: str | None,
    message_type: str,
    visibility: str,
    content: str,
) -> Message:
    """Create a message tied to an event or RSVP."""
    message = Message(
        event=event,
        rsvp=rsvp,
        author_id=author_id,
        message_type=message_type,
        visibility=visibility,
        content=content,
        created_at=_now(),
    )
    session.add(message)
    session.flush()
    return message

def set_rsvp_status(
    session: Session,
    *,
    rsvp: RSVP,
    status: str,
) -> RSVP:
    """Update the approval status for an RSVP."""
    rsvp.approval_status = _normalize_approval(status)
    rsvp.last_modified = _now()
    session.add(rsvp)
    session.flush()
    return rsvp

def update_event(
    session: Session,
    event: Event,
    *,
    title: str,
    description: str | None,
    start_time: datetime,
    end_time: datetime | None,
    location: str | None = None,
    channel: Channel | None = None,
    admin_approval_required: bool,
    is_private: bool,
    max_attendees: int | None = None,
    rsvps_closed: bool | None = None,
    rsvp_close_at: datetime | None = None,
    update_rsvp_close_at: bool = False
) -> Event:
    """Update an existing event."""
    normalized_start = to_naive_utc(start_time)
    normalized_end = to_naive_utc(end_time)
    event.title = title
    event.description = description
    event.start_time = normalized_start
    event.end_time = normalized_end
    event.location = location
    event.is_private = is_private
    event.admin_approval_required = admin_approval_required
    event.channel = channel
    event.max_attendees = max_attendees
    if rsvps_closed is not None:
        event.rsvps_closed = rsvps_closed
    if update_rsvp_close_at:
        event.rsvp_close_at = to_naive_utc(rsvp_close_at)
    event.last_modified = _now()
    session.add(event)
    session.flush()
    return event


def update_rsvp(
    session: Session,
    rsvp: RSVP,
    *,
    name: str,
    attendance_status: str,
    approval_status: str | None = None,
    pronouns: str | None,
    guest_count: int | None,
    is_private: bool,
) -> RSVP:
    """Update RSVP fields without altering approval unless provided."""
    normalized_attendance = _normalize_attendance(attendance_status)
    normalized_approval = (
        _normalize_approval(approval_status)
        if approval_status is not None
        else rsvp.approval_status
    )
    rsvp.name = name
    rsvp.attendance_status = normalized_attendance
    rsvp.approval_status = normalized_approval
    rsvp.pronouns = pronouns
    rsvp.guest_count = min(max(guest_count or 0, 0), 5)
    rsvp.is_private = bool(is_private)
    rsvp.last_modified = _now()
    session.add(rsvp)
    session.flush()
    return rsvp
