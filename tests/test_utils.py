from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from openrsvp.utils import (
    duration_between,
    get_repo_url,
    humanize_time,
    render_markdown,
    slugify,
)


def test_slugify_handles_whitespace_and_unicode():
    assert slugify("  Café au Lait  ") == "cafe-au-lait"
    assert slugify("Hello!! World??") == "hello-world"


def test_render_markdown_renders_blocks_and_inline_html():
    text = """
    # Heading

    This is **bold**, *italic*, and `code` with a [link](https://example.com).

    - Item one
    - Item two

    > Block quote
    """
    html = render_markdown(text)
    assert "<h1>Heading</h1>" in html
    assert (
        "<p>This is <strong>bold</strong>, <em>italic</em>, and <code>code</code>"
        in html
    )
    assert '<a href="https://example.com"' in html
    assert "<ul>" in html and "<li>Item one</li>" in html
    assert "<blockquote>Block quote</blockquote>" in html


def test_render_markdown_rejects_unsafe_links():
    text = "[bad](javascript:alert(1)) ok"
    html = render_markdown(text)
    assert "javascript:alert" in html  # unchanged because it is unsafe
    assert "<a" not in html  # dangerous link should not be converted


def test_humanize_time_handles_future_and_past():
    now = datetime(2024, 1, 1, 12, 0, 0)
    future = now + timedelta(days=2, hours=3)
    past = now - timedelta(seconds=10)
    assert humanize_time(future, now=now) == "in 2 days"
    assert humanize_time(past, now=now) == "moments ago"


def test_humanize_time_prefers_days_over_inexact_week():
    now = datetime(2025, 12, 1, 12, 0, 0)
    future = now + timedelta(days=11)
    assert humanize_time(future, now=now) == "in 11 days"


def test_duration_between_formats_hours_and_minutes():
    start = datetime(2024, 1, 1, 10, 0, 0)
    end = start + timedelta(hours=1, minutes=30)
    assert duration_between(start, end) == "1h 30m"


def test_get_repo_url_normalizes_github_ssh_paths(tmp_path: Path):
    config_file = tmp_path / "config"
    config_file.write_text(
        '[remote "origin"]\nurl = git@github.com:example/OpenRSVP.git\n'
    )
    assert get_repo_url(config_file) == "https://github.com/example/OpenRSVP"
