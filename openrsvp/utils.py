"""Utility helpers for OpenRSVP."""
from __future__ import annotations

import re
import unicodedata

_slug_invalid = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Return a canonical slug suitable for URLs."""
    value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    value = value.strip().lower()
    value = _slug_invalid.sub("-", value)
    value = value.strip("-")
    return value
