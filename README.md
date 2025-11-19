# OpenRSVP

OpenRSVP is a minimalist, accountless RSVP service built with FastAPI, SQLite,
and magic-link authentication. This repository provides the v0.1 single
container deployment requested in the specification.

## Features

- Magic-link authentication for root admin, event admins, and RSVP guests
- FastAPI + Jinja2 UI using Bootstrap 5 (CDN)
- SQLite storage with SQLAlchemy models
- Background decay + cleanup using APScheduler (runs hourly)
- Public/private channel system with dedicated listings at `/channel/<slug>` and
  a `/help` explainer
- Typer CLI (`openrsvp`) for server management and token rotation

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
- Channel names are slugified; reusing the same slug + visibility reuses the
  existing channel and bumps its `last_used_at`.
- Public channels show up on the homepage filter and at `/channel/<slug>`.
- Private channels never appear in lists; you must know the slug/URL to view them.
- Events inherit none of the channel visibility rules—private events stay
  private regardless of channel type.

For guidance and safety tips, share `http://localhost:8000/help` (or the
deployed host) with event creators.

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
