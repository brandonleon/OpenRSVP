"""Utility helpers for OpenRSVP."""

from __future__ import annotations

from configparser import ConfigParser
from datetime import UTC, datetime
from pathlib import Path
import html
import re
import unicodedata

from markupsafe import Markup

_slug_invalid = re.compile(r"[^a-z0-9]+")
_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_bold_pattern = re.compile(r"\*\*(.+?)\*\*")
_italic_pattern = re.compile(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)")
_code_pattern = re.compile(r"`([^`]+)`")


def utcnow() -> datetime:
    """Return a naive UTC datetime."""

    return datetime.now(UTC).replace(tzinfo=None)


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


def _sanitize_href(raw: str | None) -> str | None:
    """Return a safe URL for anchors or ``None`` when unsafe."""

    normalized = (raw or "").strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered.startswith(("http://", "https://", "mailto:")) or normalized.startswith(
        ("/", "#")
    ):
        return html.escape(normalized, quote=True)
    return None


def _render_inline(text: str) -> str:
    """Render inline Markdown (emphasis, code, and links) into sanitized HTML."""

    coded = _code_pattern.sub(lambda match: f"<code>{match.group(1)}</code>", text)
    bolded = _bold_pattern.sub(lambda match: f"<strong>{match.group(1)}</strong>", coded)
    emphasized = _italic_pattern.sub(lambda match: f"<em>{match.group(1)}</em>", bolded)

    def replace_link(match: re.Match[str]) -> str:
        href = _sanitize_href(html.unescape(match.group(2)))
        if not href:
            return match.group(0)
        label = _render_inline(match.group(1))
        return f'<a href="{href}" rel="nofollow noopener noreferrer">{label}</a>'

    return _link_pattern.sub(replace_link, emphasized)


def render_markdown(value: str | None) -> Markup:
    """Convert Markdown text into sanitized HTML safe for templates."""

    if not value:
        return Markup("")

    escaped = html.escape(value.strip())
    if not escaped:
        return Markup("")

    lines = escaped.splitlines()
    blocks: list[str] = []
    list_items: list[str] = []
    paragraph_parts: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_parts:
            paragraph = " ".join(paragraph_parts).strip()
            if paragraph:
                blocks.append(f"<p>{_render_inline(paragraph)}</p>")
            paragraph_parts.clear()

    def flush_list() -> None:
        if list_items:
            items = "".join(f"<li>{_render_inline(item)}</li>" for item in list_items)
            blocks.append(f"<ul>{items}</ul>")
            list_items.clear()

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            flush_list()
            hashes = len(stripped) - len(stripped.lstrip("#"))
            level = min(max(hashes, 1), 6)
            content = stripped[hashes:].strip()
            blocks.append(f"<h{level}>{_render_inline(content)}</h{level}>")
            continue

        normalized_line = stripped
        if stripped.startswith("&gt;"):
            normalized_line = f">{stripped[4:]}"
        if normalized_line.startswith(">"):
            flush_paragraph()
            flush_list()
            content = normalized_line.lstrip(">").strip()
            blocks.append(f"<blockquote>{_render_inline(content)}</blockquote>")
            continue

        if stripped.startswith(("- ", "* ")):
            flush_paragraph()
            list_items.append(stripped[2:].strip())
            continue

        paragraph_parts.append(stripped)

    flush_paragraph()
    flush_list()

    return Markup("\n".join(blocks))


def humanize_time(value: datetime | None, *, now: datetime | None = None) -> str:
    """Return a friendly string such as 'in 2 weeks' or '3 hours ago'."""
    if not value:
        return ""
    now = now or utcnow()
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
