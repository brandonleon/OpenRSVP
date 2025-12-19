# OpenRSVP – Agent Guide

This file provides coding‑agent rules for working in the OpenRSVP repository.

## Build / Test / Lint
- Run server: `uv run python main.py`
- Run all tests: `uv run pytest`
- Run single test: `uv run pytest tests/test_api.py::test_event_create`
- Lint/format: `uv run ruff check` and `uv run ruff format`
- Type check: `uv run pyright`

## Code Style
- Use `ruff` formatting defaults; no manual alignment.
- Imports: stdlib → third‑party → local; use explicit imports.
- Types: prefer full type hints; avoid `Any` unless required.
- Naming: snake_case for functions/vars; PascalCase for classes; UPPER for constants.
- Error handling: raise `HTTPException` in API layer; avoid silent failures.
- Database: use CRUD helpers; never bypass SQLModel session patterns.
- Templates: keep Jinja logic minimal; move logic into Python.
- No global state; use dependency injection where possible.
- Follow existing FastAPI + SQLModel patterns; avoid new frameworks.

## Agent Rules
- Preserve privacy model: never expose tokens.
- Avoid modifying RSVP content except where tests require.
- Keep patches minimal and consistent with surrounding code.

## Release Flow
- Update `CHANGELOG.md` under `Unreleased` on `develop`.
- Open PR from `develop` → `main` and merge it.
- Pull `main`, tag the merge commit (e.g., `vX.Y.Z`), and push the tag.
- Create the release from the tag.
