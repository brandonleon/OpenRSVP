# OpenRSVP – Claude Code Guide

OpenRSVP is a self-hostable, **account-free RSVP platform** built with FastAPI + SQLite. Authentication is entirely magic-link-based (secret tokens in URLs — no sessions, no user accounts).

---

## Project Layout

```
openrsvp/
├── api.py          # All FastAPI routes + handlers (~3400 lines, the core)
├── cli.py          # Typer CLI (runserver, seed-data, decay, config, etc.)
├── models.py       # SQLAlchemy ORM: Meta, Event, RSVP, Channel, Message
├── crud.py         # DB operations (use these instead of raw queries)
├── config.py       # Settings dataclass; loaded from openrsvp.toml + env vars
├── database.py     # SQLAlchemy engine, SessionLocal, get_session()
├── storage.py      # DB init, Alembic runner, root token management
├── decay.py        # Score-based exponential decay cycle logic
├── scheduler.py    # APScheduler wiring (decay + VACUUM)
├── ip_filter.py    # IP allowlist for /metrics endpoint
├── seed.py         # Faker-based dev data seeder
├── web.py          # Help section route handlers
├── utils/
│   ├── __init__.py # slugify, render_markdown, utcnow, humanize_time, etc.
│   └── ics.py      # iCalendar (.ics) export
├── templates/      # Jinja2 HTML templates
│   └── partials/   # HTMX partial fragments (admin RSVP cards)
├── static/         # CSS, JS, PWA icons
└── alembic/        # Migration scripts
tests/
├── conftest.py     # In-memory SQLite fixtures, scheduler no-ops
├── test_api.py     # Integration tests (hundreds of cases, ~39KB)
├── test_crud.py    # CRUD unit tests
├── test_ics.py     # iCal export tests
├── test_metrics.py # Metrics + IP filter tests
├── test_migrations.py
├── test_storage.py
└── test_utils.py
```

---

## Commands

```bash
# Development server (auto-reload)
uv run python main.py --dev
# or
openrsvp runserver --dev

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_api.py::test_event_create

# Lint
uv run ruff check

# Format
uv run ruff format

# Type check
uv run pyright

# Seed fake data (dev)
openrsvp seed-data

# Run decay manually
openrsvp decay

# View/set config
openrsvp config show
openrsvp config set decay_factor 0.90
```

---

## Architecture & Key Patterns

### Authentication
- Three token roles: **root admin** (server-wide, stored in `meta` table), **event admin** (per-event, stored in `events.admin_token`), **RSVP guest** (per-RSVP, stored in `rsvps.rsvp_token`).
- All tokens are `secrets.token_urlsafe(32)` strings stored **as plaintext** in SQLite. Never expose them in logs or error messages.
- No sessions, no cookies, no OAuth.

### Database Access
- **Always use CRUD helpers in `crud.py`** for DB operations — do not write raw SQLAlchemy queries in `api.py` unless the helper doesn't exist.
- In FastAPI route handlers, inject `db: Session = Depends(get_db)`.
- In CLI/scheduler code, use `with get_session() as session:`.
- Database: SQLite at `data/openrsvp.db`. For tests: in-memory SQLite via `conftest.py`.

### Response Format (Dual-Mode)
Many routes return **HTML by default** and **JSON when `Accept: application/json`** is present. The pattern is:
```python
if "application/json" in request.headers.get("accept", ""):
    return JSONResponse(...)
return templates.TemplateResponse(...)
```
Preserve this pattern on new routes that need both web and API access.

### Score-Based Decay
- Every `Event` and `Channel` has a `score` float (starts at `initial_event_score`, default 100.0).
- APScheduler runs `decay.py` hourly: `score = score * decay_factor ^ elapsed_days`.
- Below `hide_threshold` (default 10.0) → hidden from listings.
- Below `delete_threshold` (default 2.0) after `delete_after_days` (default 30) → eligible for deletion.
- Events that haven't ended yet are protected from deletion when `protect_upcoming_events=true`.

### HTMX Partials
Admin RSVP management uses HTMX. Partial templates are in `templates/partials/`. Routes returning partials check for `HX-Request` header.

### Config System (Three Layers)
`openrsvp.toml` → `OPENRSVP_<KEY>` environment variables → hardcoded defaults in `config.py`. When adding a new config knob, add it to the `Settings` dataclass with a default.

---

## Database Schema (Summary)

| Table | Key Columns |
|-------|-------------|
| `meta` | `key` (PK), `value` — stores root token, metrics IPs, etc. |
| `channels` | `id`, `slug` (unique), `name`, `visibility`, `score` |
| `events` | `id`, `admin_token` (unique), `channel_id` (FK), `title`, `start_time`, `score`, `is_private`, `admin_approval_required`, `rsvps_closed`, `max_attendees` |
| `rsvps` | `id`, `event_id` (FK), `rsvp_token` (unique), `name`, `attendance_status` (yes/no/maybe), `status` (pending/approved/rejected), `guest_count` (0-5) |
| `messages` | `id`, `event_id` (FK), `rsvp_id` (FK), `message_type`, `visibility` (public/attendee/admin), `content` |

Migrations are managed with **Alembic** (`openrsvp alembic/`). Always create a new migration when modifying models:
```bash
# Generate migration
uv run alembic revision --autogenerate -m "description"
# Apply
openrsvp upgrade-db
```

---

## Code Style

- **Formatter/Linter:** `ruff` (defaults). Run before committing.
- **Imports:** stdlib → third-party → local. Explicit imports only.
- **Types:** Full type hints everywhere. Avoid `Any` unless unavoidable.
- **Naming:** `snake_case` functions/variables, `PascalCase` classes, `UPPER_CASE` constants.
- **Errors:** Raise `HTTPException` in API layer. No silent failures.
- **Templates:** Keep Jinja2 logic minimal — move logic into Python.
- **No global state.** Use FastAPI dependency injection.
- **Follow existing FastAPI + SQLAlchemy patterns.** Don't introduce new frameworks.

---

## Testing

- Tests use **in-memory SQLite** (`conftest.py`) — fast, isolated, no disk state.
- `clean_database` fixture (autouse) drops and recreates all tables before each test.
- The APScheduler is monkeypatched to no-ops in tests.
- Add tests to the appropriate file (`test_api.py` for routes, `test_crud.py` for CRUD helpers, etc.).
- Use `TestClient` from FastAPI with the `httpx` backend.

---

## Privacy & Security Rules

- **Never expose tokens** in responses, logs, or error messages.
- **Never bypass the magic-link auth model** — don't add session/cookie auth.
- Location details of approval-required events are hidden from the public event page until the RSVP is approved.
- IP filtering for `/metrics` is enforced via `ip_filter.py` — don't bypass it.
- The `is_private` flag on both events and RSVPs must be respected in all listing queries.

---

## Deployment

**Development:**
```bash
docker compose -f docker-compose.dev.yml up --build
```

**Production:**
```bash
docker compose up --build -d
openrsvp upgrade-container --prod
```

TLS is handled by nginx + Certbot (Let's Encrypt). Certificates live in `deploy/certs/`.

---

## Current Version

`0.17.0` (see `CHANGELOG.md`). Follow [Keep a Changelog](https://keepachangelog.com/) format for changelog entries. The `develop` branch is used for active development; PRs merge to `main`.
