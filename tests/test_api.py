from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from openrsvp import api, database
from openrsvp.crud import create_event, ensure_channel
from openrsvp.models import Event, Message, RSVP
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


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _make_event(
    session, *, max_attendees: int | None = None, title: str = "Capacity Test"
):
    start = utcnow().replace(microsecond=0)
    event = create_event(
        session,
        title=title,
        description="",
        start_time=start,
        end_time=None,
        location="Somewhere",
        channel=None,
        is_private=False,
        admin_approval_required=False,
        max_attendees=max_attendees,
    )
    session.commit()
    return event


def _create_pending_rsvp(
    session,
    event: Event,
    *,
    name: str = "Pending Guest",
    attendance_status: str = "yes",
    is_private: bool = False,
) -> RSVP:
    rsvp = crud.create_rsvp(
        session,
        event=event,
        name=name,
        attendance_status=attendance_status,
        pronouns=None,
        guest_count=0,
        is_private=is_private,
        approval_status="pending",
    )
    session.flush()
    return rsvp


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
    crud.create_rsvp(
        session,
        event=event,
        name="Public Guest",
        attendance_status="yes",
        pronouns=None,
        guest_count=0,
        is_private=False,
    )
    crud.create_rsvp(
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


def test_list_public_events_excludes_private_and_paginates(client):
    session = database.SessionLocal()
    now = utcnow().replace(microsecond=0)
    public_event = create_event(
        session,
        title="Open To All",
        description="",
        start_time=now + timedelta(days=1),
        end_time=None,
        location="Somewhere",
        channel=None,
        is_private=False,
    )
    create_event(
        session,
        title="Secret Event",
        description="",
        start_time=now + timedelta(days=2),
        end_time=None,
        location="Somewhere",
        channel=None,
        is_private=True,
    )
    session.commit()

    response = client.get("/api/v1/events?per_page=1&page=1")
    assert response.status_code == 200
    data = response.json()
    event_ids = [evt["id"] for evt in data["events"]]
    assert public_event.id in event_ids
    assert data["pagination"]["per_page"] == 1
    assert data["pagination"]["total_events"] == 1

    session.close()


def test_list_public_events_supports_date_filters(client):
    session = database.SessionLocal()
    now = utcnow().replace(microsecond=0)
    early = create_event(
        session,
        title="Early",
        description="",
        start_time=now + timedelta(days=5),
        end_time=None,
        location="Here",
        channel=None,
        is_private=False,
    )
    create_event(
        session,
        title="Later",
        description="",
        start_time=now + timedelta(days=12),
        end_time=None,
        location="There",
        channel=None,
        is_private=False,
    )
    session.commit()

    start_after = _iso(now + timedelta(days=2))
    start_before = _iso(now + timedelta(days=9))
    response = client.get(
        f"/api/v1/events?start_after={start_after}&start_before={start_before}"
    )
    assert response.status_code == 200
    data = response.json()
    returned_titles = {evt["title"] for evt in data["events"]}
    assert returned_titles == {early.title}  # only early falls within bounds
    assert data["filters"]["start_after"].startswith(start_after)
    assert data["filters"]["start_before"].startswith(start_before)

    session.close()


def test_list_public_events_rejects_invalid_date_range(client):
    start_after = _iso(utcnow())
    start_before = _iso(utcnow() - timedelta(days=1))
    response = client.get(
        f"/api/v1/events?start_after={start_after}&start_before={start_before}"
    )
    assert response.status_code == 400


def test_list_public_events_per_page_limit_enforced(client):
    response = client.get("/api/v1/events?per_page=200")
    assert response.status_code == 422


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
        f"/api/v1/events/{event.id}/rsvps/{rsvp_id}/approve", headers=headers
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


def test_admin_approve_htmx_returns_partial_update(client):
    session = database.SessionLocal()
    event = create_event(
        session,
        title="HTMX Event",
        description="",
        start_time=utcnow().replace(microsecond=0),
        end_time=None,
        location="Somewhere",
        channel=None,
        is_private=False,
        admin_approval_required=True,
    )
    first = _create_pending_rsvp(session, event, name="HTMX Guest")
    _create_pending_rsvp(session, event, name="Pending Friend")
    session.commit()

    url = f"/e/{event.id}/admin/{event.admin_token}/rsvp/{first.id}/approve"
    response = client.post(url, headers={"HX-Request": "true"})

    assert response.status_code == 200
    body = response.text
    assert 'data-approval-status="approved"' in body
    assert "Pending (1)" in body  # second guest still pending
    assert 'hx-swap-oob="outerHTML"' in body
    assert "HTMX Guest" in body
    session.close()


def test_admin_approve_non_htmx_redirects(client):
    session = database.SessionLocal()
    event = create_event(
        session,
        title="Classic Event",
        description="",
        start_time=utcnow().replace(microsecond=0),
        end_time=None,
        location="Somewhere",
        channel=None,
        is_private=False,
        admin_approval_required=True,
    )
    rsvp = _create_pending_rsvp(session, event, name="Classic Guest")
    session.commit()

    url = f"/e/{event.id}/admin/{event.admin_token}/rsvp/{rsvp.id}/approve"
    response = client.post(url, follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"].startswith(
        f"/e/{event.id}/admin/{event.admin_token}"
    )
    session.close()


def test_api_rsvp_creation_logs_admin_event_message(client):
    session = database.SessionLocal()
    event = _make_event(session, title="Message Logging Event")
    event_id = event.id
    session.close()

    payload = {
        "name": "Message Watcher",
        "attendance_status": "yes",
        "guest_count": 1,
        "note": "Can't wait",
        "is_private_rsvp": False,
    }
    response = client.post(f"/api/v1/events/{event_id}/rsvps", json=payload)
    assert response.status_code == 201

    verify_session = database.SessionLocal()
    messages = (
        verify_session.query(Message)
        .filter(
            Message.event_id == event_id,
            Message.message_type == "rsvp_created",
        )
        .all()
    )
    assert len(messages) == 1
    message = messages[0]
    assert message.visibility == "admin"
    assert message.rsvp_id is not None
    assert message.rsvp is not None
    assert message.rsvp.name == payload["name"]
    assert "RSVP created" in message.content
    verify_session.close()


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
        f"/api/v1/events/{event.id}/rsvps/{rsvp_id}/reject",
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


def test_unlimited_event_allows_yes(client):
    session = database.SessionLocal()
    event = _make_event(session, max_attendees=None, title="Unlimited Event")

    for i in range(4):
        resp = client.post(
            f"/api/v1/events/{event.id}/rsvps",
            json={
                "name": f"Unlimited Guest {i}",
                "attendance_status": "yes",
                "guest_count": 0,
                "is_private_rsvp": False,
            },
        )
        assert resp.status_code == 201

    session.refresh(event)
    assert event.max_attendees is None
    assert event.yes_count == 4
    session.close()


def test_event_cap_blocks_extra_yes(client):
    session = database.SessionLocal()
    event = _make_event(session, max_attendees=3, title="Capped Event")

    for i in range(3):
        resp = client.post(
            f"/api/v1/events/{event.id}/rsvps",
            json={
                "name": f"Capped Guest {i}",
                "attendance_status": "yes",
                "guest_count": 0,
                "is_private_rsvp": False,
            },
        )
        assert resp.status_code == 201

    fourth = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Overflow Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert fourth.status_code == 400
    assert fourth.json() == {
        "error": "EventFull",
        "message": "This event has reached its maximum number of attendees.",
    }

    session.refresh(event)
    assert event.yes_count == 3
    session.close()


def test_event_cap_counts_guest_party_size(client):
    session = database.SessionLocal()
    event = _make_event(session, max_attendees=3, title="Guest Counted Event")

    primary = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Party Of Three",
            "attendance_status": "yes",
            "guest_count": 2,
            "is_private_rsvp": False,
        },
    )
    assert primary.status_code == 201

    overflow = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Overflow Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert overflow.status_code == 400
    assert overflow.json() == api.EVENT_FULL_ERROR

    session.refresh(event)
    assert event.yes_count == 3
    session.close()


def test_maybe_and_no_allowed_when_full(client):
    session = database.SessionLocal()
    event = _make_event(session, max_attendees=1, title="Full Event")

    first = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "First Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert first.status_code == 201

    maybe_resp = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Maybe Guest",
            "attendance_status": "maybe",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert maybe_resp.status_code == 201

    no_resp = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "No Guest",
            "attendance_status": "no",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert no_resp.status_code == 201

    session.refresh(event)
    assert event.yes_count == 1
    session.close()


def test_html_rsvp_submission_blocked_when_closed(client):
    session = database.SessionLocal()
    event = _make_event(session)
    event.rsvps_closed = True
    session.commit()
    response = client.post(
        f"/e/{event.id}/rsvp",
        data={
            "name": "Blocked Guest",
            "attendance_status": "yes",
            "pronouns": "",
            "guest_count": 0,
            "note": "",
            "timezone_offset_minutes": "0",
        },
    )
    assert response.status_code == 403
    assert "closed RSVPs" in response.text
    session.close()


def test_api_rsvp_submission_blocked_when_closed(client):
    session = database.SessionLocal()
    event = _make_event(session)
    event.rsvps_closed = True
    session.commit()
    response = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "API Blocked",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert response.status_code == 403
    assert response.json() == api.RSVPS_CLOSED_ERROR
    session.close()


def test_auto_close_flip_occurs_on_access(client):
    session = database.SessionLocal()
    past = utcnow() - timedelta(hours=1)
    event = create_event(
        session,
        title="Auto Close",
        description="",
        start_time=utcnow(),
        end_time=None,
        location="Space",
        channel=None,
        is_private=False,
        rsvps_closed=False,
        rsvp_close_at=past,
    )
    session.commit()
    response = client.get(f"/e/{event.id}")
    assert response.status_code == 200
    session.refresh(event)
    assert event.rsvps_closed is True
    assert event.rsvp_close_at is None
    session.close()


def test_maybe_to_yes_blocked_when_full(client):
    session = database.SessionLocal()
    event = _make_event(session, max_attendees=1, title="Maybe to Yes Block")

    primary = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Primary",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert primary.status_code == 201

    secondary = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Secondary",
            "attendance_status": "maybe",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert secondary.status_code == 201
    second_token = secondary.json()["rsvp"]["rsvp_token"]

    upgrade = client.patch(
        f"/api/v1/events/{event.id}/rsvps/self",
        headers={"Authorization": f"Bearer {second_token}"},
        json={"attendance_status": "yes"},
    )
    assert upgrade.status_code == 400
    assert upgrade.json() == {
        "error": "EventFull",
        "message": "This event has reached its maximum number of attendees.",
    }
    session.refresh(event)
    other = [r for r in event.rsvps if r.name == "Secondary"][0]
    assert other.attendance_status == "maybe"
    session.close()


def test_admin_force_accept_bypasses_cap(client):
    session = database.SessionLocal()
    event = _make_event(session, max_attendees=1, title="Force Accept Event")

    first = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "First Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert first.status_code == 201

    forced = client.post(
        f"/api/v1/events/{event.id}/rsvps?force_accept=true",
        headers={"Authorization": f"Bearer {event.admin_token}"},
        json={
            "name": "Forced Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert forced.status_code == 201

    session.refresh(event)
    assert event.yes_count == 2
    force_messages = (
        session.query(Message)
        .filter(Message.event_id == event.id, Message.message_type == "force_accept")
        .all()
    )
    assert force_messages
    session.close()


def test_yes_to_no_frees_slot(client):
    session = database.SessionLocal()
    event = _make_event(session, max_attendees=1, title="Free Slot Event")

    initial = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "First Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert initial.status_code == 201
    initial_token = initial.json()["rsvp"]["rsvp_token"]

    downgrade = client.patch(
        f"/api/v1/events/{event.id}/rsvps/self",
        headers={"Authorization": f"Bearer {initial_token}"},
        json={"attendance_status": "no"},
    )
    assert downgrade.status_code == 200

    replacement = client.post(
        f"/api/v1/events/{event.id}/rsvps",
        json={
            "name": "Replacement Guest",
            "attendance_status": "yes",
            "guest_count": 0,
            "is_private_rsvp": False,
        },
    )
    assert replacement.status_code == 201

    session.refresh(event)
    assert event.yes_count == 1
    session.close()
