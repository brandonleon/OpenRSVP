# Issue 15 – Web Push Notifications (10-day plan)

Repo: `brandonleon/OpenRSVP`  
Issue: https://github.com/brandonleon/OpenRSVP/issues/15  
Target: implement Web Push for **owners + attendees** while keeping OpenRSVP **accountless**.

## Principles / Constraints

- No users, no login, no sessions required for identity.
- Owner identity = event admin magic link (`/e/{event_id}/admin/{admin_token}`).
- Attendee identity = RSVP magic link (`/e/{event_id}/rsvp/{rsvp_token}`).
- Push subscriptions are **per-device** and stored against either:
  - `event_id` (owner subscriptions) **or**
  - `rsvp_id` (attendee subscriptions)
- When events/RSVPs are deleted (manual or scheduled cleanup), prune associated push subscriptions; optionally attempt a final “pruned” notification (best-effort) before revoking.
- Notification permission prompts only on explicit click.
- Payloads stay minimal: `{title, body, url}`.
- Sending push must not slow user flows: use `BackgroundTasks`.
- Never expose/store admin or RSVP tokens in the DB; only store push subscription data.

## Planned Architecture (MVP)

### Database

`push_subscriptions`

- `id` (PK)
- `event_id` nullable FK → `events.id` (owner subscriptions)
- `rsvp_id` nullable FK → `rsvps.id` (attendee subscriptions)
- `endpoint` (unique-ish per identity; used for idempotency)
- `p256dh`, `auth`
- `user_agent` (optional)
- `created_at`
- `revoked_at` (nullable; set on unsubscribe or dead endpoints)

Constraints / indexes:

- XOR constraint: exactly one of `event_id` / `rsvp_id` is non-null
- Index on `event_id` and `rsvp_id` for fanout
- Index (or uniqueness) on `(event_id, endpoint)` and `(rsvp_id, endpoint)` for idempotent upserts

### Backend endpoints

- `GET /push/public-key`
- `POST /push/subscribe`
- `POST /push/unsubscribe`

Auth approach:

- Prefer `Authorization: Bearer …` to avoid query-param token leakage.
- Owner subscribe/unsubscribe: bearer token must match `event.admin_token` (or root token rules if applicable).
- Attendee subscribe/unsubscribe: bearer token must match `rsvp.rsvp_token`.

### Service worker

- Serve at `/sw.js` (scope `/`) so it can control `/e/...` routes.
- `push` handler: show notification with title/body and attach `url` in `data`.
- `notificationclick` handler: open/focus `url` (fallback to `/`).

### Frontend UI (per-device opt-in)

- Event admin page: “Enable notifications” / “Notifications enabled on this device” toggle.
- RSVP created + RSVP manage pages: opt-in/out for updates to that event via that RSVP.
- Hide controls if not supported (`serviceWorker`, `PushManager`, `Notification`).

### Notification triggers (MVP)

Owner receives:

- RSVP created (Yes/Maybe)
- RSVP enters pending (including “re-enters pending” cases)
- Event updated (title/time/location/date)
- Event canceled

Attendee receives:

- RSVP approved
- RSVP rejected
- Event updated (only if RSVP approved)
- Event canceled

System cleanup/prune:

- If an event/RSVP is removed by scheduled cleanup, send a best-effort final notification (“This event/RSVP was pruned; you’ll receive no further notifications.”) and then revoke/delete associated subscriptions.

Event canceled semantics (MVP):

- If cancellation = deletion, dispatch notifications **before** delete.
- (Optional later) add explicit `canceled` state.

## Working Agreement (how we’ll work nightly)

- Keep each “night” to one coherent, reviewable slice.
- After each slice: run formatting + at least the most relevant tests.
- Prefer minimal, incremental changes over large refactors.

Common commands:

- Run server: `uv run python main.py`
- Tests: `uv run pytest`
- Lint/format: `uv run ruff check` and `uv run ruff format`
- Type check: `uv run pyright`

## 10-Day / 10-Session Implementation Plan

Timebox guideline: ~60–120 minutes/session.

### Day 1 — Kickoff + design finalization

- [ ] Confirm final schema fields + constraints for `push_subscriptions`
- [ ] Confirm endpoint auth strategy (bearer-only vs allow fallback)
- [ ] Confirm “event canceled” semantics for MVP (delete vs new status)
- [ ] Decide idempotency behavior (recommended: upsert by identity+endpoint)
- Output: finalized design notes (update this file if needed)

Validation:
- [ ] No code changes required (or only doc updates)

### Day 2 — VAPID config + `GET /push/public-key`

- [ ] Add config/env plumbing for:
  - [ ] `VAPID_PUBLIC_KEY`
  - [ ] `VAPID_PRIVATE_KEY`
  - [ ] `VAPID_SUBJECT`
- [ ] Add `GET /push/public-key` endpoint
- Output: backend returns public key when configured

Validation:
- [ ] `uv run ruff check` / `uv run ruff format`
- [ ] `uv run pytest`

### Day 3 — DB table + Alembic migration

- [ ] Add `PushSubscription` model
- [ ] Add Alembic migration for `push_subscriptions`
- [ ] Add XOR constraint and useful indexes
- Output: database supports storing/revoking subscriptions

Validation:
- [ ] Migration applies in dev
- [ ] `uv run pytest`

### Day 4 — Push delivery helper (internal)

- [ ] Add `pywebpush` dependency
- [ ] Implement push send helper (minimal payload + TTL)
- [ ] Implement “auto-revoke on 404/410”
- Output: internal function can send + revokes dead endpoints

Validation:
- [ ] Unit-test helper logic (mock webpush call)
- [ ] `uv run pytest`

### Day 5 — Subscribe/unsubscribe API

- [ ] Implement `POST /push/subscribe`
  - [ ] Owner path: authorized by admin bearer token, stores `event_id`
  - [ ] Attendee path: authorized by RSVP bearer token, stores `rsvp_id`
  - [ ] Idempotent per identity+endpoint; clear `revoked_at` on re-subscribe
- [ ] Implement `POST /push/unsubscribe` (mark `revoked_at`)
- Output: backend can persist subscriptions per device

Validation:
- [ ] Request/response shapes stable
- [ ] `uv run pytest`

### Day 6 — Service worker (`/sw.js`)

- [ ] Add SW served at `/sw.js` with scope `/`
- [ ] Implement `push` and `notificationclick` handlers safely
- Output: browser can register SW and display notifications

Validation:
- [ ] Manual test in browser (localhost secure context)
- [ ] Ensure SW doesn’t break other pages

### Day 7 — Frontend shared push client

- [ ] Add JS module to:
  - [ ] register SW
  - [ ] detect support
  - [ ] request permission only on click
  - [ ] subscribe/unsubscribe and POST to backend
  - [ ] reflect “enabled on this device” state
- Output: reusable `initPushControls()` (or similar) callable on relevant pages

Validation:
- [ ] Manual test opt-in/out from a dummy page (or existing pages behind feature flags)

### Day 8 — Owner UI integration (event admin page)

- [ ] Add “Enable notifications” UI on event admin page
- [ ] Wire to `event_id` owner subscriptions (authorized by admin token)
- Output: owner can opt-in/out per device from admin page

Validation:
- [ ] Manual test opt-in/out + verify DB rows update

### Day 9 — Attendee UI integration (RSVP created + manage pages)

- [ ] Add attendee opt-in UI after RSVP creation
- [ ] Add attendee opt-in UI on RSVP manage page
- [ ] Wire to `rsvp_id` attendee subscriptions (authorized by RSVP token)
- Output: attendee can opt-in/out per device via RSVP magic link

Validation:
- [ ] Manual test with a fresh RSVP + revisit link

### Day 10 — Triggers + docs + tests pass

- [ ] Wire owner triggers:
  - [ ] RSVP created / pending
  - [ ] Event updated
  - [ ] Event canceled
- [ ] Wire attendee triggers:
  - [ ] RSVP approved/rejected
  - [ ] Event updated (approved RSVPs only)
  - [ ] Event canceled
- [ ] Wire cleanup behavior:
  - [ ] When scheduled cleanup deletes events/RSVPs, send best-effort “pruned” notification and revoke/delete subscriptions
- [ ] Ensure push is sent via `BackgroundTasks`
- [ ] Update docs (privacy, HTTPS requirement, disabling, limitations)
- [ ] Add/extend tests for subscription selection + dispatch calls
- Output: end-to-end MVP meets issue acceptance criteria

Validation:
- [ ] `uv run ruff check` / `uv run ruff format`
- [ ] `uv run pytest`
- [ ] `uv run pyright` (best-effort; fix only issues introduced here)

## Parking Lot / Later (explicitly out of MVP unless we decide otherwise)

- Rate-limiting/throttling (beyond basic “don’t spam” checks)
- Rich notification payloads (images, actions)
- Explicit event cancel state (vs delete semantics)
- Additional channels (email/SMS/Discord) beyond the dispatcher interface
