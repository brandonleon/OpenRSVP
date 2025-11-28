"""CRUD helpers for events, RSVPs, and channels."""

from __future__ import annotations

import secrets
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .models import Channel, Event, Message, RSVP
from .utils import slugify, utcnow

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


def get_public_channels(session: Session, limit: int | None = None) -> list[Channel]:
    stmt = (
        select(Channel)
        .where(Channel.visibility == "public")
        .order_by(Channel.score.desc(), Channel.name.asc())
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


def create_event(
    session: Session,
    *,
    title: str,
    description: str | None,
    start_time: datetime,
    end_time: datetime | None,
    location: str | None,
    channel: Channel | None,
    admin_approval_required: bool = False,
    is_private: bool = False,
) -> Event:
    """Create and persist a new event."""
    event = Event(
        admin_token=secrets.token_urlsafe(32),
        is_private=is_private,
        admin_approval_required=admin_approval_required,
        title=title,
        description=description,
        start_time=start_time,
        end_time=end_time,
        location=location,
        score=settings.initial_event_score,
        channel=channel,
    )
    session.add(event)
    session.flush()
    return event


def update_event(
    session: Session,
    event: Event,
    *,
    title: str,
    description: str | None,
    start_time: datetime,
    end_time: datetime | None,
    location: str | None,
    channel: Channel | None,
    admin_approval_required: bool,
    is_private: bool,
) -> Event:
    """Update an existing event."""
    event.title = title
    event.description = description
    event.start_time = start_time
    event.end_time = end_time
    event.location = location
    event.is_private = is_private
    event.admin_approval_required = admin_approval_required
    event.channel = channel
    event.last_modified = _now()
    session.add(event)
    session.flush()
    return event


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


def _normalize_attendance(status: str | None) -> str:
    normalized = (status or "").strip().lower() or "yes"
    return normalized if normalized in VALID_ATTENDANCE_STATUSES else "maybe"


def _normalize_approval(status: str | None) -> str:
    normalized = (status or "").strip().lower() or "approved"
    return normalized if normalized in VALID_APPROVAL_STATUSES else "pending"


def create_rsvp(
    session: Session,
    *,
    event: Event,
    name: str,
    attendance_status: str,
    approval_status: str | None = None,
    pronouns: str | None,
    guest_count: int | None,
    is_private: bool = False,
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
