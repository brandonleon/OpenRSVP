from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from openrsvp import api, database
from openrsvp.crud import create_event


@pytest.fixture()
def client(monkeypatch):
    """FastAPI test client with the scheduler disabled."""

    monkeypatch.setattr(api, "start_scheduler", lambda: None)
    monkeypatch.setattr(api, "stop_scheduler", lambda: None)
    with TestClient(api.app) as test_client:
        yield test_client


def _make_event(*, start: datetime, end: datetime) -> str:
    session = database.SessionLocal()
    event = create_event(
        session,
        title="Calendar Test",
        description="Line one\nLine two",
        start_time=start,
        end_time=end,
        location="Test Venue",
        channel=None,
        is_private=False,
    )
    session.commit()
    session.close()
    return event.id


def test_ics_endpoint_serves_calendar_file(client):
    start = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    event_id = _make_event(start=start, end=end)

    response = client.get(f"/api/v1/events/{event_id}/event.ics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/calendar")
    assert f'filename="event_{event_id}.ics"' in response.headers["content-disposition"]

    body = response.text
    assert "BEGIN:VCALENDAR" in body
    assert "BEGIN:VEVENT" in body
    assert f"UID:{event_id}@openrsvp" in body
    assert "SUMMARY:Calendar Test" in body
    assert "DESCRIPTION:Line one\\\\nLine two" in body
    assert "LOCATION:Test Venue" in body
    assert "DTSTART:" in body
    assert "DTEND:" in body


def test_ics_dates_are_normalized_to_utc(client):
    start = datetime(2024, 6, 1, 9, 0, tzinfo=timezone(timedelta(hours=-5)))
    end = start + timedelta(hours=2)
    event_id = _make_event(start=start, end=end)

    response = client.get(f"/api/v1/events/{event_id}/event.ics")
    assert response.status_code == 200
    body = response.text
    # 09:00 -05:00 should convert to 14:00Z
    assert "DTSTART:20240601T140000Z" in body
    assert "DTEND:20240601T160000Z" in body
