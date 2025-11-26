from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from openrsvp import api, database
from openrsvp.crud import create_event, ensure_channel
from openrsvp.models import Event
from openrsvp.utils import utcnow
from openrsvp import crud


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


def test_channel_routes_respect_visibility(client):
    session = database.SessionLocal()
    public_channel = ensure_channel(session, name="Public Lounge", visibility="public")
    private_channel = ensure_channel(
        session, name="Secret Cellar", visibility="private"
    )
    session.commit()

    public_response = client.get(f"/channel/{public_channel.slug}")
    assert public_response.status_code == 200
    assert f"{public_channel.name}" in public_response.text

    public_on_private_route = client.get(f"/channel/p/{public_channel.slug}")
    assert public_on_private_route.status_code == 404

    private_response = client.get(f"/channel/p/{private_channel.slug}")
    assert private_response.status_code == 200
    assert f"{private_channel.name}" in private_response.text

    private_on_public_route = client.get(f"/channel/{private_channel.slug}")
    assert private_on_public_route.status_code == 404

    session.close()


def test_public_event_in_private_channel_hides_channel_on_event_page(client):
    session = database.SessionLocal()
    channel = ensure_channel(session, name="Hidden Den", visibility="private")
    start = utcnow().replace(microsecond=0)
    event = create_event(
        session,
        title="Open Jam",
        description="Music night",
        start_time=start,
        end_time=None,
        location="Basement",
        channel=channel,
        is_private=False,
    )
    session.commit()

    event_page = client.get(f"/e/{event.id}")
    assert event_page.status_code == 200
    assert "Open Jam" in event_page.text
    assert "Hidden Den" not in event_page.text

    admin_page = client.get(f"/e/{event.id}/admin/{event.admin_token}")
    assert admin_page.status_code == 200
    assert "Hidden Den" in admin_page.text

    session.close()


def test_private_rsvp_hidden_from_public_list(client):
    session = database.SessionLocal()
    channel = ensure_channel(session, name="Mixed", visibility="public")
    start = utcnow().replace(microsecond=0)
    event = create_event(
        session,
        title="Mixer",
        description="desc",
        start_time=start,
        end_time=None,
        location="Somewhere",
        channel=channel,
        is_private=False,
    )
    public = crud.create_rsvp(
        session,
        event=event,
        name="Public Guest",
        attendance_status="yes",
        pronouns=None,
        guest_count=0,
        is_private=False,
    )
    private = crud.create_rsvp(
        session,
        event=event,
        name="Private Guest",
        attendance_status="yes",
        pronouns=None,
        guest_count=0,
        is_private=True,
    )
    session.commit()

    event_page = client.get(f"/e/{event.id}")
    assert event_page.status_code == 200
    assert "Public Guest" in event_page.text
    assert "Private Guest" not in event_page.text
    assert "+ 1 private" in event_page.text

    admin_page = client.get(f"/e/{event.id}/admin/{event.admin_token}")
    assert admin_page.status_code == 200
    assert "Public Guest" in admin_page.text
    assert "Private Guest" in admin_page.text
    assert "Private" in admin_page.text  # badge exists

    session.close()


def test_event_page_counts_public_and_private_party_sizes(client):
    session = database.SessionLocal()
    event = create_event(
        session,
        title="Headcount",
        description="",
        start_time=utcnow().replace(microsecond=0),
        end_time=None,
        location="Somewhere",
        channel=None,
        is_private=False,
    )
    # public RSVP: party of 3 (1 + 2 guests)
    crud.create_rsvp(
        session,
        event=event,
        name="Public One",
        attendance_status="yes",
        pronouns=None,
        guest_count=2,
        is_private=False,
    )
    # private RSVP: party of 2 (1 + 1 guest)
    crud.create_rsvp(
        session,
        event=event,
        name="Private One",
        attendance_status="yes",
        pronouns=None,
        guest_count=1,
        is_private=True,
    )
    session.commit()

    event_page = client.get(f"/e/{event.id}")
    assert event_page.status_code == 200
    assert "3 public people (1 RSVP" in event_page.text
    assert "+ 2 private people (1 RSVP" in event_page.text

    session.close()


def test_pending_rsvp_hidden_until_approved(client):
    session = database.SessionLocal()
    event = create_event(
        session,
        title="Approval Needed",
        description="",
        start_time=utcnow().replace(microsecond=0),
        end_time=None,
        location="Somewhere",
        channel=None,
        is_private=False,
        admin_approval_required=True,
    )
    session.commit()

    create_resp = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Pending Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert create_resp.status_code == 201
    created_rsvp = create_resp.json()["rsvp"]
    assert created_rsvp["approval_status"] == "pending"

    public_resp = client.get(f"/api/v1/events/{event.id}")
    public_event = public_resp.json()["event"]
    counts = public_event["rsvp_counts"]
    assert counts["public"] == 0
    assert counts["public_party_size"] == 0
    assert public_event["rsvps"] == []

    headers = {"Authorization": f"Bearer {event.admin_token}"}
    admin_list = client.get(f"/api/v1/events/{event.id}/rsvps", headers=headers)
    assert admin_list.status_code == 200
    admin_rsvp = admin_list.json()["rsvps"][0]
    assert admin_rsvp["approval_status"] == "pending"

    rsvp_id = admin_rsvp["id"]
    approve_resp = client.post(
        f"/events/{event.id}/rsvps/{rsvp_id}/approve", headers=headers
    )
    assert approve_resp.status_code == 200
    approved_rsvp = approve_resp.json()["rsvp"]
    assert approved_rsvp["approval_status"] == "approved"
    message_types = {m["message_type"] for m in approved_rsvp["messages"]}
    assert "rsvp_status_change" in message_types

    public_after = client.get(f"/api/v1/events/{event.id}")
    event_after = public_after.json()["event"]
    counts_after = event_after["rsvp_counts"]
    assert counts_after["public"] == 1
    assert counts_after["public_party_size"] == 1
    assert len(event_after["rsvps"]) == 1
    assert event_after["rsvps"][0]["name"] == "Pending Guest"

    session.close()


def test_rejection_adds_attendee_message_and_keeps_public_hidden(client):
    session = database.SessionLocal()
    event = create_event(
        session,
        title="Approval Needed",
        description="",
        start_time=utcnow().replace(microsecond=0),
        end_time=None,
        location="Somewhere",
        channel=None,
        is_private=False,
        admin_approval_required=True,
    )
    session.commit()

    create_resp = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Rejectable Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert create_resp.status_code == 201
    rsvp_payload = create_resp.json()["rsvp"]
    rsvp_token = rsvp_payload["rsvp_token"]

    headers = {"Authorization": f"Bearer {event.admin_token}"}
    admin_list = client.get(f"/api/v1/events/{event.id}/rsvps", headers=headers)
    rsvp_id = admin_list.json()["rsvps"][0]["id"]

    reason = "Invite list is full"
    reject_resp = client.post(
        f"/events/{event.id}/rsvps/{rsvp_id}/reject",
        headers=headers,
        json={"reason": reason},
    )
    assert reject_resp.status_code == 200
    rejected_rsvp = reject_resp.json()["rsvp"]
    assert rejected_rsvp["approval_status"] == "rejected"
    message_by_type = {
        m["message_type"]: m["content"] for m in rejected_rsvp["messages"]
    }
    assert message_by_type["rejection_reason"] == reason
    assert "RSVP marked as rejected" in message_by_type["rsvp_status_change"]

    attendee_view = client.get(
        f"/api/v1/events/{event.id}/rsvps/self",
        headers={"Authorization": f"Bearer {rsvp_token}"},
    )
    assert attendee_view.status_code == 200
    attendee_messages = attendee_view.json()["rsvp"]["messages"]
    attendee_types = {m["message_type"]: m["content"] for m in attendee_messages}
    assert attendee_types["rejection_reason"] == reason

    public_resp = client.get(f"/api/v1/events/{event.id}")
    public_counts = public_resp.json()["event"]["rsvp_counts"]
    assert public_counts["public"] == 0
    assert public_counts["public_party_size"] == 0

    session.close()
