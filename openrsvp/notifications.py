"""Discord webhook notification system for OpenRSVP events."""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import and_, select

from .config import settings
from .database import get_session
from .models import Event, EventNotification

logger = logging.getLogger("uvicorn.error")

# Tolerance window: how many minutes either side of the target time we accept.
# Set to half the check interval so no window is missed.
TOLERANCE_MINUTES = settings.discord_notify_check_minutes / 2


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _notification_key(minutes_before: int) -> str:
    return f"{minutes_before}m"


def _already_sent(session, event_id: str, key: str) -> bool:
    row = session.execute(
        select(EventNotification).where(
            and_(
                EventNotification.event_id == event_id,
                EventNotification.notification_key == key,
            )
        )
    ).scalar_one_or_none()
    return row is not None


def _record_sent(session, event_id: str, key: str) -> None:
    record = EventNotification(
        event_id=event_id,
        notification_key=key,
    )
    session.add(record)


def _format_time(dt: datetime, iana_tz: str | None = None) -> str:
    """Format a UTC datetime as a human-friendly string, optionally in a local timezone."""
    if iana_tz:
        try:
            tz = ZoneInfo(iana_tz)
            local = dt.replace(tzinfo=timezone.utc).astimezone(tz)
            return local.strftime("%A, %B %-d at %-I:%M %p %Z")
        except ZoneInfoNotFoundError:
            pass
    return dt.strftime("%A, %B %-d at %-I:%M %p UTC")


def _build_embed(event: Event, minutes_before: int) -> dict:
    if minutes_before == 0:
        title = f"Event starting now: {event.title}"
        color = 0x57F287  # green
    elif minutes_before < 60:
        title = f"Event in {minutes_before} minutes: {event.title}"
        color = 0xFEE75C  # yellow
    elif minutes_before < 120:
        title = f"Event in 1 hour: {event.title}"
        color = 0xFEE75C
    else:
        hours = minutes_before // 60
        title = f"Event in {hours} hours: {event.title}"
        color = 0x5865F2  # blurple

    fields = []
    fields.append(
        {
            "name": "Starts",
            "value": _format_time(event.start_time, event.timezone),
            "inline": True,
        }
    )
    if event.end_time:
        fields.append(
            {
                "name": "Ends",
                "value": _format_time(event.end_time, event.timezone),
                "inline": True,
            }
        )
    if event.location:
        fields.append(
            {
                "name": "Location",
                "value": event.location,
                "inline": False,
            }
        )

    yes_count = event.yes_count
    if yes_count:
        fields.append(
            {
                "name": "Attendees",
                "value": str(yes_count),
                "inline": True,
            }
        )

    embed: dict = {
        "title": title,
        "color": color,
        "fields": fields,
    }
    if event.description:
        # Truncate long descriptions to fit Discord's 4096-char limit.
        desc = event.description[:300]
        if len(event.description) > 300:
            desc += "…"
        embed["description"] = desc

    return embed


def send_discord_notification(webhook_url: str, event: Event, minutes_before: int) -> bool:
    """POST a Discord embed to the given webhook URL. Returns True on success."""
    embed = _build_embed(event, minutes_before)
    payload = json.dumps({"embeds": [embed]}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "OpenRSVP/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            if status in (200, 204):
                return True
            logger.warning(
                "Discord webhook returned unexpected status %s for event %s",
                status,
                event.id,
            )
            return False
    except urllib.error.HTTPError as exc:
        logger.error(
            "Discord webhook HTTP error %s for event %s: %s",
            exc.code,
            event.id,
            exc.reason,
        )
        return False
    except Exception as exc:
        logger.error(
            "Discord webhook request failed for event %s: %s",
            event.id,
            exc,
        )
        return False


def run_notification_check() -> None:
    """Check for upcoming events and send Discord notifications as needed."""
    intervals = settings.discord_notify_intervals_list
    if not intervals:
        return

    now = _utcnow()

    with get_session() as session:
        # Load all upcoming events that have a webhook configured.
        upcoming = session.execute(
            select(Event).where(
                and_(
                    Event.discord_webhook_url.isnot(None),
                    Event.discord_webhook_url != "",
                    Event.start_time > now - timedelta(minutes=1),
                )
            )
        ).scalars().all()

        for event in upcoming:
            webhook_url = event.discord_webhook_url
            start = event.start_time
            minutes_until = (start - now).total_seconds() / 60

            for interval in intervals:
                key = _notification_key(interval)
                if _already_sent(session, event.id, key):
                    continue

                # Check if we're within the tolerance window of this interval.
                diff = abs(minutes_until - interval)
                if diff <= TOLERANCE_MINUTES:
                    logger.info(
                        "Sending Discord notification '%s' for event %s (%s)",
                        key,
                        event.id,
                        event.title,
                    )
                    sent = send_discord_notification(webhook_url, event, interval)
                    if sent:
                        _record_sent(session, event.id, key)
