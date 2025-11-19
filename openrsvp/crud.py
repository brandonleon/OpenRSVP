"""CRUD helpers for events, RSVPs, and channels."""

from __future__ import annotations

import secrets
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Channel, Event, RSVP
from .utils import slugify

CHANNEL_VISIBILITIES = {"public", "private"}


def _now() -> datetime:
    return datetime.utcnow()


def get_channel_by_slug(session: Session, slug: str) -> Channel | None:
    normalized = (slug or "").strip().lower()
    if not normalized:
        return None
    stmt = select(Channel).where(Channel.slug == normalized)
    return session.scalars(stmt).first()


def get_public_channels(session: Session) -> list[Channel]:
    stmt = (
        select(Channel)
        .where(Channel.visibility == "public")
        .order_by(Channel.score.desc(), Channel.name.asc())
    )
    return session.scalars(stmt).all()


def touch_channel(channel: Channel, *, timestamp: datetime | None = None) -> None:
    channel.last_used_at = timestamp or _now()


def ensure_channel(
    session: Session,
    *,
    name: str,
    visibility: str,
) -> Channel:
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
        score=100.0,
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
    is_private: bool = False,
) -> Event:
    event = Event(
        admin_token=secrets.token_urlsafe(32),
        is_private=is_private,
        title=title,
        description=description,
        start_time=start_time,
        end_time=end_time,
        location=location,
        score=100.0,
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
    is_private: bool,
) -> Event:
    event.title = title
    event.description = description
    event.start_time = start_time
    event.end_time = end_time
    event.location = location
    event.is_private = is_private
    event.channel = channel
    event.last_modified = _now()
    session.add(event)
    session.flush()
    return event


def create_rsvp(
    session: Session,
    *,
    event: Event,
    name: str,
    status: str,
    pronouns: str | None,
    guest_count: int | None,
    notes: str | None,
) -> RSVP:
    rsvp = RSVP(
        event=event,
        rsvp_token=secrets.token_urlsafe(32),
        name=name,
        status=status,
        pronouns=pronouns,
        guest_count=guest_count or 0,
        notes=notes,
        score=100.0,
    )
    session.add(rsvp)
    session.flush()
    return rsvp


def update_rsvp(
    session: Session,
    rsvp: RSVP,
    *,
    name: str,
    status: str,
    pronouns: str | None,
    guest_count: int | None,
    notes: str | None,
) -> RSVP:
    rsvp.name = name
    rsvp.status = status
    rsvp.pronouns = pronouns
    rsvp.guest_count = min(max(guest_count or 0, 0), 5)
    rsvp.notes = notes
    rsvp.last_modified = _now()
    session.add(rsvp)
    session.flush()
    return rsvp
