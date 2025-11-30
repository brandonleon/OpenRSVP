"""iCalendar (.ics) helpers."""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openrsvp.models import Event


_tag_pattern = re.compile(r"<[^>]+>")


def _ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _format_utc(dt: datetime) -> str:
    """Format a datetime as an RFC5545 UTC timestamp."""

    return _ensure_utc(dt).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _escape_text(value: str | None) -> str:
    """Escape text for ICS fields and strip any HTML tags."""

    if not value:
        return ""
    # Remove HTML tags and entities, then escape reserved characters.
    stripped = _tag_pattern.sub("", html.unescape(value))
    normalized = stripped.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    for line in normalized.split("\n"):
        trimmed = line.lstrip()
        if trimmed.startswith("#"):
            trimmed = trimmed.lstrip("#").lstrip()
        if trimmed.startswith(">"):
            trimmed = trimmed.lstrip(">").lstrip()
        if trimmed.startswith(("- ", "* ")):
            trimmed = trimmed[2:].lstrip()
        cleaned_lines.append(trimmed)
    normalized = "\n".join(cleaned_lines)
    escaped = (
        normalized.replace("\\", "\\\\")
        .replace(";", r"\;")
        .replace(",", r"\,")
        .replace("\n", "\\\\n")
    )
    return escaped


def generate_ics(event: Event, *, now: datetime | None = None) -> str:
    """Return ICS text for an event."""

    dtstamp = _format_utc(now or datetime.now(UTC))
    start = _format_utc(event.start_time)
    end_source = event.end_time or event.start_time
    end = _format_utc(end_source)
    summary = _escape_text(event.title)
    description = _escape_text(event.description)
    location = _escape_text(event.location)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//OpenRSVP//EN",
        "BEGIN:VEVENT",
        f"UID:{event.id}@openrsvp",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART:{start}",
        f"DTEND:{end}",
        f"SUMMARY:{summary}",
        f"DESCRIPTION:{description}",
        f"LOCATION:{location}",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(lines) + "\r\n"
