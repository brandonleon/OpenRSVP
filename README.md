# OpenRSVP

OpenRSVP is a minimalist, accountless RSVP service built with FastAPI, SQLite,
and magic-link authentication. This repository provides the v0.1 single
container deployment requested in the specification.

## Features

- Magic-link authentication for root admin, event admins, and RSVP guests
- Device-side helpers: copy-to-clipboard toasts and optional local storage so browsers can reopen admin/RSVP magic links (clearable in the UI)
- Catppuccin themes (Latte, Mocha, Frappé, Macchiato) with per-browser preference and system fallback
- FastAPI + Jinja2 UI using Bootstrap 5 (CDN)
- SQLite storage with SQLAlchemy models
- Private RSVP toggle to hide a guest from public lists while keeping the organizer’s view intact
- Background decay + cleanup using APScheduler (runs hourly)
- Public/private channel system with dedicated listings at `/channel/<slug>` and
  a `/help` explainer
- Typer CLI (`openrsvp`) for server management and token rotation
- Optional end-time support for events with automatic timezone conversion

## Getting Started

```bash
git clone <repo>
cd OpenRSVP
uv venv
source .venv/bin/activate
uv pip install -e .
openrsvp runserver
```

Visit `http://localhost:8000` to access the UI. Use the CLI to retrieve the root
admin token:

```bash
openrsvp admin-token
```

## Docker

Build and run the container:

```bash
docker build -t openrsvp .
docker run -p 8000:8000 openrsvp
```

The Docker image is based on `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
and syncs dependencies via `uv sync` (the `.venv` it creates is placed on
`PATH`). The container entrypoint starts `openrsvp runserver`, which launches
FastAPI and the decay scheduler.

## Channels

- Channel names and visibility (public/private) are chosen when creating an event.
- Channel names are slugified (lowercased, ASCII-only, non-alphanumeric
  collapsed to hyphens, with leading/trailing hyphens removed); reusing the
  same slug + visibility reuses the existing channel and bumps its
  `last_used_at`.
- Public channels show up on the homepage filter and at `/channel/<slug>`.
- Private channels never appear in lists; you must know the slug/URL to view
  them (`/channel/p/<slug>`).
- Events inherit none of the channel visibility rules—private events stay
  private regardless of channel type.
- Public events inside private channels hide the channel badge to avoid leaking
  the slug.

For guidance and safety tips, share `http://localhost:8000/help` (or the
deployed host) with event creators.

## Magic Links, Privacy, and Preferences

- Guests receive a one-time magic link after submitting an RSVP; it is also
  stored locally (when possible) so the browser can surface “Edit your RSVP” on
  the event page. Organizers’ admin links are treated the same way.
- Local storage is per-device only. Use the Preferences menu or “Forget this
  key” buttons on event/RSVP pages to clear saved links.
- Guests can mark an RSVP private so it hides from the public attendee list;
  organizers still see it. Organizers cannot edit RSVP contents but can delete
  entries on request.
- Event owners can delete an entire event from the admin page or via
  `DELETE /api/v1/events/{event_id}` using `Authorization: Bearer <admin-token>`.
  Deletion wipes all RSVPs and cannot be undone.
- The Preferences menu includes a Catppuccin theme picker. The chosen palette
  persists locally and falls back to your system light/dark setting if unset.

## Manual Decay & Vacuum

The Typer CLI exposes the decay cycle so you can trigger cleanup jobs outside
of the scheduler:

```bash
openrsvp decay          # decay without vacuum
openrsvp decay --vacuum # decay followed by SQLite VACUUM
```

`--vacuum` is useful after large data churn or manual deletions, since VACUUM
reclaims disk space at the cost of briefly locking the database.

## Generating Fake Data

For local testing you can seed the SQLite database with synthetic channels,
events, and RSVPs using the Typer CLI. This command relies on the Faker dev
dependency.

```bash
openrsvp seed-data --channels 6 --max-events 4 --extra-events 2 --max-rsvps 5 --private-percent 15
```

Each run creates new channels (public or private), fills them with events, and
attaches randomized RSVP records. The `--extra-events` flag optionally creates
events that are not attached to any channel, `--max-rsvps` caps RSVP counts per
event, and `--private-percent` controls what portion of events are marked
private.

## Running Tests

Use `uv` so dev dependencies install on demand:

```bash
PYTHONPATH=. UV_CACHE_DIR=.uv-cache uv run pytest
```

`PYTHONPATH=.` makes the local `openrsvp` package importable without an editable
install. The `UV_CACHE_DIR` override keeps cache files in the repo instead of
the default user cache (helpful on locked-down environments).

You can also run the same defaults via the CLI:

```bash
uv run main.py test              # or: openrsvp test
```
