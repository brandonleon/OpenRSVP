# OpenRSVP Feature Backlog

Prioritized by user value and implementation complexity relative to the existing architecture.

---

## P1 — High Priority

High value, fits existing architecture naturally, no major new dependencies.

- [ ] **Waitlist system** — Extend the existing `max_attendees` + approval workflow. Add a `waitlisted` attendance status to RSVPs; auto-promote the next waitlisted guest when an approved attendee cancels. Builds on `crud.py` RSVP helpers and the existing approval flow.

- [ ] **CSV export** — Admin-only route to download attendee list as a `.csv` file. Simple route in `api.py`, Python stdlib `csv` module, no new dependencies. Include name, attendance status, guest count, and approval status.

- [ ] **Event duplication** — Clone an existing event from the admin page. Pre-fills the creation form (or directly inserts a copy) using existing CRUD helpers; strips the old admin token and generates a new one. Useful for recurring manual events.

- [ ] **Webhook support** — POST to a configurable URL on RSVP lifecycle events (created, approved, rejected, cancelled). Store the webhook URL in the `meta` table. Use `httpx` (already a dev dependency — promote to runtime) for outbound requests.

- [ ] **Full-text event search** — Add SQLite FTS5 virtual table (built-in, no new deps) indexing `events.title` and `events.description`. Wire up a `/search` route with dual HTML/JSON response mode.

---

## P2 — Medium Priority

High value but more complex, or requires new dependencies or schema changes.

- [ ] **Email notifications** — SMTP config via `openrsvp config set smtp_*`. Send transactional emails on approve/reject/update. Use stdlib `smtplib` or optional `aiosmtplib` for async sending. Requires new config keys in `Settings` and an Alembic migration if storing notification preferences.

- [ ] **QR code check-in** — Add a `checked_in` boolean column to `rsvps` (Alembic migration required). Admin scans a QR code encoding the RSVP token URL to toggle check-in status. A `qrcode.min.js` library is already referenced in static assets — reuse it for display.

- [ ] **RSVP reminder messages** — APScheduler job (extend `scheduler.py`) that creates `Message` records N days before an event's `start_time`. Reuses the existing `messages` table and the scheduler infrastructure. N configurable via `openrsvp config`.

- [ ] **RSS/Atom feed for channels** — New route `/channel/{slug}/feed.xml` returning an Atom feed of upcoming public events in a channel. Pure Python XML generation (`xml.etree.ElementTree`), no new dependencies.

- [ ] **iCal subscription URL** — Live-updating `.ics` for an entire channel (not just a single event). Extends existing `utils/ics.py` to aggregate multiple events. New route `/channel/{slug}/calendar.ics`.

---

## P3 — Lower Priority

Nice-to-have features, significant scope, or niche use cases.

- [ ] **Custom RSVP questions** — New `rsvp_questions` and `rsvp_answers` tables (Alembic migration). Admin defines questions per event; guests answer on the RSVP form. Requires a form builder UI in templates.

- [ ] **Recurring events** — New `event_series` table linking related event instances. Date math for generating occurrences. Significant UI and CRUD work; touches decay/deletion logic.

- [ ] **Event templates** — Store JSON event config snapshots in the `meta` table. Template picker on the event creation form. Allows admins to reuse common event structures quickly.

- [ ] **Event co-hosts** — New `event_admins` table associating additional tokens with an event. Multiple admin token management and revocation. Requires rethinking the single-token admin auth model.

- [ ] **Bulk RSVP import** — CSV upload by event admin; generate an `rsvp_token` per row and optionally deliver via email. Depends on email notification feature for delivery.

- [ ] **Audit log** — Append-only log table recording admin actions (approve, reject, delete, config change). Extends the `messages` table pattern or adds a dedicated `audit_log` table. Useful for accountability on shared instances.

- [ ] **SMTP/email config CLI** — `openrsvp config set smtp_host`, `smtp_port`, `smtp_user`, etc. Depends on the email notifications feature. Add a `openrsvp config test-email` command to verify SMTP settings.
