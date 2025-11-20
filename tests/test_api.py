from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from openrsvp import api, database
from openrsvp.crud import create_event, ensure_channel
from openrsvp.models import Event
from openrsvp.utils import utcnow


@pytest.fixture()
def client(monkeypatch):
    """FastAPI test client with the scheduler disabled."""

    monkeypatch.setattr(api, "start_scheduler", lambda: None)
    monkeypatch.setattr(api, "stop_scheduler", lambda: None)
    with TestClient(api.app) as test_client:
        yield test_client


def _datetime_str(dt: datetime) -> str:
    return dt.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M")


def test_submit_event_creates_event_and_channel(client):
    start = utcnow().replace(microsecond=0)
    end = start + timedelta(hours=1)
    payload = {
        "title": "Launch Party",
        "description": "All welcome",
        "start_time": _datetime_str(start),
        "end_time": _datetime_str(end),
        "timezone_offset_minutes": "0",
        "location": "HQ",
        "channel_name": "Announcements",
        "channel_visibility": "public",
        "is_private": "true",
    }
    response = client.post("/events", data=payload)
    assert response.status_code == 200

    session = database.SessionLocal()
    events = session.query(Event).all()
    assert len(events) == 1
    created = events[0]
    assert created.title == "Launch Party"
    assert created.is_private is True
    assert created.channel is not None
    assert created.channel.name == "Announcements"
    assert created.channel.visibility == "public"
    session.close()


def test_event_admin_can_change_and_remove_channel(client):
    session = database.SessionLocal()
    channel = ensure_channel(session, name="Existing", visibility="public")
    start = utcnow().replace(microsecond=0)
    event = create_event(
        session,
        title="Weekly Sync",
        description="Discuss progress",
        start_time=start,
        end_time=None,
        location="Online",
        channel=channel,
        is_private=False,
    )
    session.commit()

    admin_path = f"/e/{event.id}/admin/{event.admin_token}"
    payload = {
        "title": event.title,
        "description": event.description,
        "start_time": _datetime_str(event.start_time),
        "end_time": "",
        "timezone_offset_minutes": "0",
        "location": event.location,
        "channel_name": "Secret Club",
        "channel_visibility": "private",
        "is_private": "true",
    }
    response = client.post(admin_path, data=payload)
    assert response.status_code == 200
    session.refresh(event)
    assert event.channel is not None
    assert event.channel.name == "Secret Club"
    assert event.channel.visibility == "private"
    assert event.is_private is True

    removal_payload = {
        "title": event.title,
        "description": event.description,
        "start_time": _datetime_str(event.start_time),
        "end_time": "",
        "timezone_offset_minutes": "0",
        "location": event.location,
        "channel_name": "",
        "channel_visibility": "public",
    }
    response = client.post(admin_path, data=removal_payload)
    assert response.status_code == 200
    session.refresh(event)
    assert event.channel is None
    assert event.is_private is False
    session.close()
