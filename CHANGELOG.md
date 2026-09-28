# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to SemVer.

## [Unreleased]

## [0.18.0a1] - 2026-09-28

### Added

- Repeating events: choose daily, weekly, biweekly, or monthly recurrence and a
  count (2–52) when creating an event to generate a series of independent
  occurrences, each with its own RSVP page and admin link.
- `EventSeries` model and `event_series` table (migration `0003_repeating_events`)
  to group occurrences; each `Event` gains a nullable `series_id` FK.
- Series admin page at `/series/<id>/admin/<token>` listing all occurrences with
  RSVP counts and quick-manage links.
- "Part of a series →" badge on individual event admin pages linking back to
  the series overview.
- Documentation page `docs/repeating-events.html` covering recurrence rules,
  the series admin page, per-occurrence independence, and data model details.
- Per-event guest allowance (`max_guests`, 0–5, default 5): hosts choose how many
  extra guests each RSVP may bring; 0 turns guests off. Available in the create
  and admin forms and the JSON API. Migration `0004_event_max_guests` backfills
  existing events with 5, so current behavior and stored RSVPs are unchanged.
  Existing RSVPs keep their guests until the guest edits them.

### Changed

- Capacity ("spots filled") and the public attendee badge now count only
  "yes" RSVPs that are not rejected; pending RSVPs still hold their spot.
  The public badge reads "N public people attending" and ignores maybe/no.
- The justfile `dev` recipe now runs `main.py runserver --dev`.

### Fixed

- Opening an event admin page with the root admin token no longer returns a 500
  (`get_session()` now uses an independent session instead of closing the
  request's thread-local session).
- Rejected RSVPs no longer consume event capacity.
- Existing "yes" guests can edit their RSVP when an event is over capacity
  (e.g. after a force-accept) as long as their party size does not grow.

### Removed

## [0.16.0] - 2025-12-19

### Added

- Deployment guide and updated docs navigation, plus API messaging details.
- Channel discovery page with search, sorting, and pagination.
- "My Events" and "My RSVPs" pages driven by local saved magic links.
- Event share button with native share sheet support and clipboard fallback.
- Location gating for approval-required events in the API and ICS export.
- Dev server reload flag (`openrsvp runserver --dev`).
- Docker upgrade options for host port and data directory overrides.
- Security probe script, Kubernetes diagrams, and scaling notes.
- Expanded API and privacy-focused test coverage.

### Changed

- Hide event location and unapproved RSVP counts from public API/listing views
  unless the requester is authorized.
- API event payloads now omit private channel metadata unless authorized.
- Home page channel selector limited to top-ranked public channels.
- Share and channel UI copy updated across templates and docs.

### Fixed

- Public API privacy leaks for private RSVPs and private channels.
