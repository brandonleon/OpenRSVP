"""Tests for metrics endpoint and IP filtering."""

from __future__ import annotations

import ipaddress
from datetime import timedelta
from unittest.mock import MagicMock, Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from openrsvp.api import app
from openrsvp.crud import create_event, create_rsvp, ensure_channel
from openrsvp.database import SessionLocal
from openrsvp.ip_filter import (
    check_ip_allowed,
    get_client_ip,
    parse_cidr_list,
    require_metrics_access,
    validate_cidr,
)
from openrsvp.models import Channel, Event, Meta, RSVP
from openrsvp.storage import init_db
from openrsvp.utils import utcnow


@pytest.fixture
def client(monkeypatch):
    """Test client fixture with scheduler disabled."""
    from openrsvp import api

    monkeypatch.setattr(api, "start_scheduler", lambda: None)
    monkeypatch.setattr(api, "stop_scheduler", lambda: None)
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def db():
    """Database session fixture."""
    from openrsvp.database import SessionLocal as DBSessionLocal

    session = DBSessionLocal()
    yield session
    session.close()


class TestIPValidation:
    """Test IP address validation and parsing."""

    def test_validate_cidr_ipv4_valid(self):
        """Valid IPv4 CIDR ranges."""
        assert validate_cidr("192.168.1.0/24")
        assert validate_cidr("10.0.0.0/8")
        assert validate_cidr("172.16.0.0/12")
        assert validate_cidr("127.0.0.1/32")

    def test_validate_cidr_ipv6_valid(self):
        """Valid IPv6 CIDR ranges."""
        assert validate_cidr("2001:db8::/32")
        assert validate_cidr("::1/128")
        assert validate_cidr("fe80::/10")

    def test_validate_cidr_invalid(self):
        """Invalid CIDR notation."""
        assert not validate_cidr("not.an.ip")
        assert not validate_cidr("192.168.1.0/33")  # Invalid prefix length
        assert not validate_cidr("192.168.1.256/24")  # Invalid octet
        assert not validate_cidr("")
        # Note: "192.168.1.0" without prefix is valid (treated as /32 by ipaddress)

    def test_parse_cidr_list_single(self):
        """Parse single CIDR range."""
        networks = parse_cidr_list("192.168.1.0/24")
        assert len(networks) == 1
        assert isinstance(networks[0], ipaddress.IPv4Network)
        assert str(networks[0]) == "192.168.1.0/24"

    def test_parse_cidr_list_multiple(self):
        """Parse multiple CIDR ranges."""
        networks = parse_cidr_list("192.168.1.0/24,10.0.0.0/8,::1/128")
        assert len(networks) == 3
        assert isinstance(networks[0], ipaddress.IPv4Network)
        assert isinstance(networks[1], ipaddress.IPv4Network)
        assert isinstance(networks[2], ipaddress.IPv6Network)

    def test_parse_cidr_list_with_spaces(self):
        """Parse CIDR list with spaces."""
        networks = parse_cidr_list("192.168.1.0/24, 10.0.0.0/8 , ::1/128")
        assert len(networks) == 3

    def test_parse_cidr_list_empty(self):
        """Empty CIDR list."""
        assert parse_cidr_list("") == []
        assert parse_cidr_list("   ") == []

    def test_parse_cidr_list_invalid(self):
        """Invalid CIDR in list raises ValueError."""
        with pytest.raises(ValueError, match="Invalid CIDR"):
            parse_cidr_list("192.168.1.0/24,not.valid")

    def test_check_ip_allowed_ipv4(self):
        """Check IPv4 address against ranges."""
        ranges = parse_cidr_list("192.168.1.0/24,10.0.0.0/8")

        assert check_ip_allowed("192.168.1.100", ranges)
        assert check_ip_allowed("10.5.5.5", ranges)
        assert not check_ip_allowed("172.16.0.1", ranges)
        assert not check_ip_allowed("8.8.8.8", ranges)

    def test_check_ip_allowed_ipv6(self):
        """Check IPv6 address against ranges."""
        ranges = parse_cidr_list("2001:db8::/32,::1/128")

        assert check_ip_allowed("2001:db8::1", ranges)
        assert check_ip_allowed("::1", ranges)
        assert not check_ip_allowed("2001:db9::1", ranges)

    def test_check_ip_allowed_empty_ranges(self):
        """Empty range list denies all."""
        assert not check_ip_allowed("192.168.1.1", [])

    def test_check_ip_allowed_invalid_ip(self):
        """Invalid IP address is denied."""
        ranges = parse_cidr_list("192.168.1.0/24")
        assert not check_ip_allowed("not.an.ip", ranges)
        assert not check_ip_allowed("", ranges)

    def test_get_client_ip_direct(self):
        """Get client IP from direct connection."""
        request = Mock()
        request.headers.get.return_value = None
        request.client.host = "192.168.1.100"

        assert get_client_ip(request) == "192.168.1.100"

    def test_get_client_ip_forwarded(self):
        """Get client IP from X-Forwarded-For header."""
        request = Mock()
        request.headers.get.return_value = "203.0.113.1, 192.168.1.1"
        request.client.host = "192.168.1.1"

        # Should return first IP (original client)
        assert get_client_ip(request) == "203.0.113.1"

    def test_get_client_ip_forwarded_single(self):
        """Get client IP from X-Forwarded-For with single IP."""
        request = Mock()
        request.headers.get.return_value = "203.0.113.1"
        request.client.host = "192.168.1.1"

        assert get_client_ip(request) == "203.0.113.1"

    def test_get_client_ip_no_client(self):
        """Fallback when no client info available."""
        request = Mock()
        request.headers.get.return_value = None
        request.client = None

        assert get_client_ip(request) == "unknown"


class TestMetricsAccess:
    """Test metrics access control."""

    def test_require_metrics_access_no_config(self, db):
        """Deny access when no IP ranges configured."""
        request = Mock()
        request.headers.get.return_value = None
        request.client.host = "127.0.0.1"

        with pytest.raises(HTTPException) as exc_info:
            require_metrics_access(request, db)

        assert exc_info.value.status_code == 403
        assert "Configure allowed IPs" in exc_info.value.detail

    def test_require_metrics_access_allowed(self, db):
        """Allow access from configured IP range."""
        # Configure allowed IP
        meta = Meta(key="metrics_allowed_ips", value="127.0.0.1/32", updated_at=utcnow())
        db.merge(meta)
        db.commit()

        request = Mock()
        request.headers.get.return_value = None
        request.client.host = "127.0.0.1"

        # Should not raise
        require_metrics_access(request, db)

    def test_require_metrics_access_denied(self, db):
        """Deny access from non-allowed IP."""
        # Configure allowed IP
        meta = Meta(key="metrics_allowed_ips", value="192.168.1.0/24", updated_at=utcnow())
        db.merge(meta)
        db.commit()

        request = Mock()
        request.headers.get.return_value = None
        request.client.host = "10.0.0.1"

        with pytest.raises(HTTPException) as exc_info:
            require_metrics_access(request, db)

        assert exc_info.value.status_code == 403
        assert "10.0.0.1" in exc_info.value.detail

    def test_require_metrics_access_invalid_config(self, db):
        """Handle invalid CIDR configuration."""
        # Configure invalid CIDR
        meta = Meta(key="metrics_allowed_ips", value="not.valid", updated_at=utcnow())
        db.merge(meta)
        db.commit()

        request = Mock()
        request.headers.get.return_value = None
        request.client.host = "127.0.0.1"

        with pytest.raises(HTTPException) as exc_info:
            require_metrics_access(request, db)

        assert exc_info.value.status_code == 403
        assert "configuration error" in exc_info.value.detail.lower()


class TestMetricsEndpoint:
    """Test /metrics endpoint."""

    def test_metrics_endpoint_denied_by_default(self, client, db):
        """Metrics endpoint denies access by default."""
        response = client.get("/metrics")
        assert response.status_code == 403

    def test_metrics_collection(self, db):
        """Test metrics collection logic directly."""
        from openrsvp.api import _collect_metrics, _format_prometheus

        # Create test data
        channel = ensure_channel(db, name="Test Channel", visibility="public")
        db.commit()

        event1 = create_event(
            db,
            title="Test Event 1",
            description="Test",
            start_time=utcnow() + timedelta(days=1),
            end_time=utcnow() + timedelta(days=1, hours=2),
            location="Test Location",
            channel=channel,
            is_private=False,
            admin_approval_required=False,
            max_attendees=None,
        )
        event2 = create_event(
            db,
            title="Test Event 2",
            description="Test",
            start_time=utcnow() + timedelta(days=2),
            end_time=utcnow() + timedelta(days=2, hours=2),
            location="Test Location",
            channel=channel,
            is_private=True,
            admin_approval_required=False,
            max_attendees=None,
        )
        db.commit()

        rsvp1 = create_rsvp(
            db,
            event=event1,
            name="Test User 1",
            attendance_status="yes",
            pronouns=None,
            guest_count=0,
            is_private=False,
            approval_status="approved",
        )
        rsvp2 = create_rsvp(
            db,
            event=event1,
            name="Test User 2",
            attendance_status="no",
            pronouns=None,
            guest_count=0,
            is_private=False,
            approval_status="approved",
        )
        db.commit()

        # Test metrics collection
        metrics = _collect_metrics(db)
        assert metrics["openrsvp_events_total"] == 2
        assert metrics["openrsvp_events_active"] == 2
        assert metrics["openrsvp_events_private"] == 1
        assert metrics["openrsvp_rsvps_total"] == 2
        assert metrics["openrsvp_rsvps_yes_total"] == 1
        assert metrics["openrsvp_channels_total"] == 1
        assert metrics["openrsvp_channels_public"] == 1
        assert metrics["openrsvp_up"] == 1

        # Test Prometheus formatting
        prometheus_output = _format_prometheus(metrics)
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output
        assert "openrsvp_up 1" in prometheus_output
        assert "openrsvp_version_info" in prometheus_output
        assert "openrsvp_events_total 2" in prometheus_output
        assert "openrsvp_events_private 1" in prometheus_output
        assert "openrsvp_rsvps_yes_total 1" in prometheus_output


class TestMetricsCLI:
    """Test metrics-config CLI commands."""

    def test_cli_list_empty(self, db, capsys):
        """List command shows empty state."""
        from openrsvp.cli import metrics_list

        # List doesn't raise SystemExit on empty
        metrics_list()

    def test_cli_set_localhost(self, db):
        """Set command with --localhost flag."""
        from openrsvp.cli import metrics_set

        metrics_set(cidrs=[], localhost=True)

        # Verify it was stored
        meta = db.query(Meta).filter(Meta.key == "metrics_allowed_ips").first()
        assert meta is not None
        assert "127.0.0.1/32" in meta.value
        assert "::1/128" in meta.value

    def test_cli_set_custom(self, db):
        """Set command with custom CIDR."""
        from openrsvp.cli import metrics_set

        metrics_set(cidrs=["192.168.1.0/24"], localhost=False)

        # Verify it was stored
        meta = db.query(Meta).filter(Meta.key == "metrics_allowed_ips").first()
        assert meta is not None
        assert meta.value == "192.168.1.0/24"

    def test_cli_set_invalid(self, db):
        """Set command rejects invalid CIDR."""
        from openrsvp.cli import metrics_set
        from click.exceptions import Exit

        with pytest.raises(Exit) as exc_info:
            metrics_set(cidrs=["not.valid"], localhost=False)

        assert exc_info.value.exit_code == 1

    def test_cli_add(self, db):
        """Add command appends to existing list."""
        from openrsvp.cli import metrics_add, metrics_set

        # Set initial value
        metrics_set(cidrs=["192.168.1.0/24"], localhost=False)

        # Add another
        metrics_add(cidr="10.0.0.0/8")

        # Verify both are present
        meta = db.query(Meta).filter(Meta.key == "metrics_allowed_ips").first()
        assert meta is not None
        assert "192.168.1.0/24" in meta.value
        assert "10.0.0.0/8" in meta.value

    def test_cli_add_duplicate(self, db):
        """Add command ignores duplicates."""
        from openrsvp.cli import metrics_add, metrics_set

        # Set initial value
        metrics_set(cidrs=["192.168.1.0/24"], localhost=False)

        # Try to add same value
        metrics_add(cidr="192.168.1.0/24")

        # Verify it's only present once
        meta = db.query(Meta).filter(Meta.key == "metrics_allowed_ips").first()
        assert meta is not None
        assert meta.value == "192.168.1.0/24"

    def test_cli_remove(self, db):
        """Remove command removes from list."""
        from openrsvp.cli import metrics_remove, metrics_set

        # Set multiple values
        metrics_set(cidrs=["192.168.1.0/24", "10.0.0.0/8"], localhost=False)

        # Remove one
        metrics_remove(cidr="192.168.1.0/24")

        # Verify only one remains
        meta = db.query(Meta).filter(Meta.key == "metrics_allowed_ips").first()
        assert meta is not None
        assert "10.0.0.0/8" in meta.value
        assert "192.168.1.0/24" not in meta.value

    def test_cli_remove_last(self, db):
        """Remove command deletes entry when removing last range."""
        from openrsvp.cli import metrics_remove, metrics_set

        # Set single value
        metrics_set(cidrs=["192.168.1.0/24"], localhost=False)

        # Remove it
        metrics_remove(cidr="192.168.1.0/24")

        # Verify entry is deleted
        meta = db.query(Meta).filter(Meta.key == "metrics_allowed_ips").first()
        assert meta is None

    def test_cli_clear(self, db):
        """Clear command removes all ranges."""
        from openrsvp.cli import metrics_clear, metrics_set

        # Set values
        metrics_set(cidrs=["192.168.1.0/24", "10.0.0.0/8"], localhost=False)

        # Clear with --yes flag
        metrics_clear(yes=True)

        # Verify entry is deleted
        meta = db.query(Meta).filter(Meta.key == "metrics_allowed_ips").first()
        assert meta is None
