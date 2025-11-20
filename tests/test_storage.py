from __future__ import annotations

from openrsvp.storage import ensure_root_token, fetch_root_token, rotate_root_token


def test_root_token_lifecycle():
    first = ensure_root_token()
    assert isinstance(first, str) and first
    assert fetch_root_token() == first
    rotated = rotate_root_token()
    assert rotated != first
    assert fetch_root_token() == rotated
