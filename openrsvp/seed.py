"""Development helpers for populating fake channels and events."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from faker import Faker
from sqlalchemy.orm import Session

from .crud import (
    CHANNEL_VISIBILITIES,
    create_event,
    create_rsvp,
    ensure_channel,
    get_channel_by_slug,
)
from .database import get_session
from .models import Channel, Event
from .storage import init_db
from .utils import slugify

_channel_suffixes = [
    "Social Club",
    "Meetup",
    "Crew",
    "Collective",
    "Society",
    "Circle",
    "Gathering",
]
_event_types = [
    "Mixer",
    "Hangout",
    "Workshop",
    "Cigar Night",
    "Field Trip",
    "Meet & Greet",
    "Dinner",
    "Discussion",
]
_pronoun_sets = [
    "she/her",
    "he/him",
    "they/them",
    "she/they",
    "he/they",
    "",
]
_rsvp_statuses = ["yes", "yes", "yes", "maybe", "maybe", "no"]


def seed_fake_data(
    *,
    channel_count: int = 5,
    max_events_per_channel: int = 3,
    extra_events: int = 0,
    max_rsvps_per_event: int = 5,
    private_percentage: int = 10,
) -> dict[str, int]:
    """Populate the SQLite database with synthetic channels and events."""
    if channel_count < 0:
        raise ValueError("channel_count must be >= 0")
    if max_events_per_channel < 1:
        raise ValueError("max_events_per_channel must be >= 1")
    if extra_events < 0:
        raise ValueError("extra_events must be >= 0")
    if max_rsvps_per_event < 0:
        raise ValueError("max_rsvps_per_event must be >= 0")
    if not 0 <= private_percentage <= 100:
        raise ValueError("private_percentage must be between 0 and 100")

    init_db()
    fake = Faker()
    stats = {"channels": 0, "events": 0, "rsvps": 0}

    with get_session() as session:
        for _ in range(channel_count):
            channel = _create_channel(session, fake)
            stats["channels"] += 1
            event_total = random.randint(1, max_events_per_channel)
            for _ in range(event_total):
                rsvps = _create_event(
                    session,
                    fake,
                    channel=channel,
                    private_percentage=private_percentage,
                    max_rsvps=max_rsvps_per_event,
                )
                stats["events"] += 1
                stats["rsvps"] += rsvps

        for _ in range(extra_events):
            rsvps = _create_event(
                session,
                fake,
                channel=None,
                private_percentage=private_percentage,
                max_rsvps=max_rsvps_per_event,
            )
            stats["events"] += 1
            stats["rsvps"] += rsvps

    return stats


def _create_channel(session: Session, fake: Faker):
    for _ in range(20):
        name = f"{fake.city()} {random.choice(_channel_suffixes)}"
        slug = slugify(name)
        if not slug:
            continue
        if get_channel_by_slug(session, slug):
            continue
        visibility = random.choice(tuple(CHANNEL_VISIBILITIES))
        return ensure_channel(session, name=name, visibility=visibility)
    raise RuntimeError("Failed to create a unique channel name")


def _create_event(
    session: Session,
    fake: Faker,
    *,
    channel: Channel | None,
    private_percentage: int,
    max_rsvps: int,
) -> int:
    start_time = _random_start_time()
    end_time = _maybe_end_time(start_time)
    description = "\n\n".join(fake.paragraphs(nb=2))
    location = fake.address().replace("\n", ", ")
    is_private = bool(channel and channel.visibility == "private")
    if not is_private:
        is_private = random.randint(1, 100) <= private_percentage
    event = create_event(
        session,
        title=_event_title(fake),
        description=description,
        start_time=start_time,
        end_time=end_time,
        location=location,
        channel=channel,
        is_private=is_private,
    )
    return _create_rsvps(session, fake, event, max_rsvps)


def _random_start_time() -> datetime:
    now = datetime.utcnow()
    day_offset = random.randint(-7, 30)
    minute_offset = random.randint(0, 23 * 60)
    return now + timedelta(days=day_offset, minutes=minute_offset)


def _event_title(fake: Faker) -> str:
    prefix = fake.city()
    event_type = random.choice(_event_types)
    return f"{prefix} {event_type}"


def _maybe_end_time(start_time: datetime) -> datetime | None:
    if random.random() < 0.3:
        return None
    duration_hours = random.randint(1, 6)
    return start_time + timedelta(hours=duration_hours)


def _create_rsvps(session: Session, fake: Faker, event: Event, max_rsvps: int) -> int:
    if max_rsvps <= 0:
        return 0
    total = random.randint(0, max_rsvps)
    for _ in range(total):
        name = fake.name_nonbinary()
        pronouns = random.choice(_pronoun_sets) or None
        status = random.choice(_rsvp_statuses)
        guest_count = max(0, random.randint(0, 3))
        notes = fake.sentence() if random.random() < 0.3 else None
        create_rsvp(
            session,
            event=event,
            name=name,
            status=status,
            pronouns=pronouns,
            guest_count=guest_count,
            notes=notes,
        )
    return total
