"""Utility helpers for OpenRSVP."""

from __future__ import annotations

from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
import re
import unicodedata

_slug_invalid = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Return a canonical slug suitable for URLs."""
    value = (
        unicodedata.normalize("NFKD", value or "")
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    value = value.strip().lower()
    value = _slug_invalid.sub("-", value)
    value = value.strip("-")
    return value


def humanize_time(value: datetime | None, *, now: datetime | None = None) -> str:
    """Return a friendly string such as 'in 2 weeks' or '3 hours ago'."""
    if not value:
        return ""
    now = now or datetime.utcnow()
    delta_seconds = (value - now).total_seconds()
    past = delta_seconds < 0
    seconds = abs(delta_seconds)

    units = [
        ("year", 365 * 24 * 3600),
        ("month", 30 * 24 * 3600),
        ("week", 7 * 24 * 3600),
        ("day", 24 * 3600),
        ("hour", 3600),
        ("minute", 60),
    ]

    amount = 0
    label = "minute"
    for name, step in units:
        value_count = int(seconds // step)
        if value_count >= 1:
            amount = value_count
            label = name
            break
    else:
        return "in moments" if not past else "moments ago"

    if amount != 1:
        label = f"{label}s"
    if past:
        return f"{amount} {label} ago"
    return f"in {amount} {label}"


def duration_between(start: datetime | None, end: datetime | None) -> str:
    """Return a short "2h 30m" style duration string."""
    if not start or not end:
        return ""
    seconds = max(int((end - start).total_seconds()), 0)
    if seconds == 0:
        return "less than 1 minute"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append("<1m")
    return " ".join(parts)


def get_repo_url(config_path: Path | None = None) -> str | None:
    """Return the repository URL derived from the local Git config.

    Prefers the origin remote. SSH-style GitHub URLs are converted to HTTPS and
    trailing ``.git`` suffixes are stripped for cleaner links.
    """

    config_path = (
        config_path or Path(__file__).resolve().parent.parent / ".git" / "config"
    )
    if not config_path.is_file():
        return None

    parser = ConfigParser()
    parser.read(config_path)

    section = 'remote "origin"'
    if not parser.has_section(section):
        return None

    raw_url = parser.get(section, "url", fallback=None)
    if not raw_url:
        return None

    cleaned = raw_url.strip()
    if cleaned.startswith("git@github.com:"):
        cleaned = f"https://github.com/{cleaned.removeprefix('git@github.com:')}"
    cleaned = cleaned.rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]

    return cleaned or None
