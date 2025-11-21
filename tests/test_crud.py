from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from openrsvp import database
from openrsvp.crud import (
    create_event,
    create_rsvp,
    ensure_channel,
    touch_channel,
    update_event,
    update_rsvp,
)
from openrsvp.utils import utcnow


@pytest.fixture()
def session():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_ensure_channel_creates_and_reuses(session):
    created = ensure_channel(session, name="Test Channel", visibility="public")
    session.commit()
    reused = ensure_channel(session, name="Test Channel", visibility="public")
    assert reused.id == created.id
    assert reused.slug == "test-channel"


def test_ensure_channel_visibility_conflict(session):
    ensure_channel(session, name="Secret", visibility="private")
    session.commit()
    with pytest.raises(ValueError):
        ensure_channel(session, name="Secret", visibility="public")


def test_touch_channel_updates_timestamp(session):
    channel = ensure_channel(session, name="Stale", visibility="public")
    session.commit()
    original = channel.last_used_at
    touch_channel(channel)
    assert channel.last_used_at >= original


def test_create_and_update_event(session):
    channel = ensure_channel(session, name="Events", visibility="public")
    session.commit()
    start = utcnow()
    end = start + timedelta(hours=2)
    event = create_event(
        session,
        title="Original",
        description="Desc",
        start_time=start,
        end_time=end,
        location="Town",
        channel=channel,
        is_private=False,
    )
    session.commit()
    new_end = end + timedelta(hours=1)
    update_event(
        session,
        event,
        title="Updated",
        description="New desc",
        start_time=start,
        end_time=new_end,
        location="New place",
        channel=None,
        is_private=True,
    )
    assert event.title == "Updated"
    assert event.description == "New desc"
    assert event.end_time == new_end
    assert event.location == "New place"
    assert event.channel is None
    assert event.is_private is True


def test_create_and_update_rsvp(session):
    channel = ensure_channel(session, name="RSVPs", visibility="public")
    start = utcnow()
    event = create_event(
        session,
        title="Party",
        description=None,
        start_time=start,
        end_time=None,
        location=None,
        channel=channel,
        is_private=False,
    )
    session.commit()
    rsvp = create_rsvp(
        session,
        event=event,
        name="Alice",
        status="yes",
        pronouns="she/her",
        guest_count=1,
        notes="See you!",
        is_private=True,
    )
    session.commit()
    update_rsvp(
        session,
        rsvp,
        name="Alice B",
        status="maybe",
        pronouns=None,
        guest_count=10,
        notes=None,
        is_private=False,
    )
    assert rsvp.name == "Alice B"
    assert rsvp.status == "maybe"
    assert rsvp.pronouns is None
    assert rsvp.guest_count == 5  # clamped at max
    assert rsvp.notes is None
    assert rsvp.is_private is False
