from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from openrsvp import database
from openrsvp.crud import (
    MAX_SERIES_OCCURRENCES,
    create_event,
    create_event_series,
    create_rsvp,
    ensure_channel,
    get_events_in_series,
    get_series_by_admin_token,
    touch_channel,
    update_event,
    update_rsvp,
    _add_months,
    _offset_datetime,
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
        admin_approval_required=True,
        is_private=True,
    )
    assert event.title == "Updated"
    assert event.description == "New desc"
    assert event.end_time == new_end
    assert event.location == "New place"
    assert event.channel is None
    assert event.admin_approval_required is True
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
        attendance_status="yes",
        pronouns="she/her",
        guest_count=1,
        is_private=True,
    )
    session.commit()
    update_rsvp(
        session,
        rsvp,
        name="Alice B",
        attendance_status="maybe",
        pronouns=None,
        guest_count=10,
        is_private=False,
    )
    assert rsvp.name == "Alice B"
    assert rsvp.attendance_status == "maybe"
    assert rsvp.approval_status == "approved"
    assert rsvp.pronouns is None
    assert rsvp.guest_count == 5  # clamped at max
    assert rsvp.is_private is False


def test_rsvp_close_controls(session):
    start = utcnow()
    close_at = start + timedelta(days=1)
    event = create_event(
        session,
        title="RSVP Window",
        description=None,
        start_time=start,
        end_time=None,
        location=None,
        channel=None,
        is_private=False,
        rsvps_closed=False,
        rsvp_close_at=None,
    )
    session.commit()
    update_event(
        session,
        event,
        title=event.title,
        description=event.description,
        start_time=start,
        end_time=None,
        location=None,
        channel=None,
        admin_approval_required=False,
        is_private=False,
        rsvps_closed=True,
        rsvp_close_at=close_at,
        update_rsvp_close_at=True,
    )
    assert event.rsvps_closed is True
    assert event.rsvp_close_at == close_at


def test_add_months_normal(session):
    dt = datetime(2024, 1, 15)
    assert _add_months(dt, 1) == datetime(2024, 2, 15)
    assert _add_months(dt, 12) == datetime(2025, 1, 15)


def test_add_months_clamps_day_at_month_end(session):
    assert _add_months(datetime(2024, 1, 31), 1) == datetime(2024, 2, 29)  # leap year
    assert _add_months(datetime(2023, 1, 31), 1) == datetime(2023, 2, 28)  # non-leap


def test_offset_datetime_all_rules():
    base = datetime(2024, 3, 1, 10, 0)
    assert _offset_datetime(base, "daily", 1) == datetime(2024, 3, 2, 10, 0)
    assert _offset_datetime(base, "weekly", 1) == datetime(2024, 3, 8, 10, 0)
    assert _offset_datetime(base, "biweekly", 1) == datetime(2024, 3, 15, 10, 0)
    assert _offset_datetime(base, "monthly", 1) == datetime(2024, 4, 1, 10, 0)
    assert _offset_datetime(base, "unknown", 5) == base  # passthrough


def test_create_event_series_weekly(session):
    start = utcnow().replace(microsecond=0)
    series, events = create_event_series(
        session,
        title="Weekly Standup",
        description=None,
        start_time=start,
        end_time=None,
        location=None,
        channel=None,
        recurrence_rule="weekly",
        recurrence_count=4,
    )
    session.commit()
    assert len(events) == 4
    for i, event in enumerate(events):
        assert event.title == "Weekly Standup"
        assert event.series_id == series.id
        assert event.start_time == start + timedelta(weeks=i)


def test_create_event_series_with_end_time_preserves_duration(session):
    start = utcnow().replace(microsecond=0)
    end = start + timedelta(hours=2)
    _, events = create_event_series(
        session,
        title="Duration Test",
        description=None,
        start_time=start,
        end_time=end,
        location=None,
        channel=None,
        recurrence_rule="daily",
        recurrence_count=3,
    )
    session.commit()
    for event in events:
        assert event.end_time is not None
        assert event.end_time - event.start_time == timedelta(hours=2)


def test_create_event_series_clamps_count(session):
    start = utcnow().replace(microsecond=0)
    _, events_min = create_event_series(
        session,
        title="Min",
        description=None,
        start_time=start,
        end_time=None,
        location=None,
        channel=None,
        recurrence_rule="weekly",
        recurrence_count=0,  # below minimum
    )
    session.commit()
    assert len(events_min) == 2

    _, events_max = create_event_series(
        session,
        title="Max",
        description=None,
        start_time=start,
        end_time=None,
        location=None,
        channel=None,
        recurrence_rule="weekly",
        recurrence_count=999,  # above maximum
    )
    session.commit()
    assert len(events_max) == MAX_SERIES_OCCURRENCES


def test_create_event_series_invalid_rule(session):
    start = utcnow().replace(microsecond=0)
    with pytest.raises(ValueError, match="Invalid recurrence rule"):
        create_event_series(
            session,
            title="Bad",
            description=None,
            start_time=start,
            end_time=None,
            location=None,
            channel=None,
            recurrence_rule="hourly",
            recurrence_count=4,
        )


def test_get_series_by_admin_token(session):
    start = utcnow().replace(microsecond=0)
    series, _ = create_event_series(
        session,
        title="Token Lookup",
        description=None,
        start_time=start,
        end_time=None,
        location=None,
        channel=None,
        recurrence_rule="weekly",
        recurrence_count=2,
    )
    session.commit()
    found = get_series_by_admin_token(session, series.admin_token)
    assert found is not None
    assert found.id == series.id

    assert get_series_by_admin_token(session, "nonexistent") is None


def test_get_events_in_series_ordered(session):
    start = utcnow().replace(microsecond=0)
    series, _ = create_event_series(
        session,
        title="Order Test",
        description=None,
        start_time=start,
        end_time=None,
        location=None,
        channel=None,
        recurrence_rule="weekly",
        recurrence_count=3,
    )
    session.commit()
    events = get_events_in_series(session, series.id)
    assert len(events) == 3
    assert events[0].start_time < events[1].start_time < events[2].start_time


def test_rsvp_pending_until_approved_when_required(session):
    channel = ensure_channel(session, name="Approval", visibility="public")
    start = utcnow()
    event = create_event(
        session,
        title="Gatekept Party",
        description=None,
        start_time=start,
        end_time=None,
        location=None,
        channel=channel,
        is_private=False,
        admin_approval_required=True,
    )
    session.commit()
    rsvp = create_rsvp(
        session,
        event=event,
        name="Pending Pal",
        attendance_status="yes",
        pronouns=None,
        guest_count=0,
        is_private=False,
    )
    assert rsvp.approval_status == "pending"

    update_rsvp(
        session,
        rsvp,
        name="Pending Pal",
        attendance_status="maybe",
        pronouns=None,
        guest_count=1,
        is_private=False,
    )
    assert rsvp.approval_status == "pending"

    update_rsvp(
        session,
        rsvp,
        name="Approved Pal",
        attendance_status="yes",
        approval_status="approved",
        pronouns=None,
        guest_count=2,
        is_private=False,
    )
    assert rsvp.approval_status == "approved"
