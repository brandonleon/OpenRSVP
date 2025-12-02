# OpenRSVP – Agents & Responsibilities (v0.1)

OpenRSVP is an accountless, privacy friendly RSVP platform. All access is gated
by capability tokens rather than accounts. The sections below describe each
agent capability and the system-wide rules that support them.

## General Notes

- Capabilities are represented only through secure tokens or magic links.
- Tokens are never exposed outside their intended scope.
- Data is ephemeral. Events and RSVPs decay automatically unless accessed.
- Magic links can be remembered locally in the browser for quick access and are clearable via the Preferences menu or per-event “Forget” actions.

## 1. Root Admin Agent

**Identity:** Holds the root admin token retrieved via
`docker exec openrsvp openrsvp admin-token`

**Authentication:** CLI only; never surfaced in the UI.

### Capabilities (Root Admin)

- View every event, including stale, hidden, or decaying entries.
- See all RSVPs (including private ones) and channel visibility settings.
- Inspect event metadata, decay scores, and RSVP records (read only).
- Access any event admin link or RSVP token.
- Trigger the decay cycle: `openrsvp decay`.
- Force delete events or rotate the root admin token.

### Restrictions (Root Admin)

- Cannot modify RSVP content on behalf of guests.
- Cannot impersonate RSVP users.
- Cannot disable or bypass the decay system.
- Cannot restore events once purged.

### Purpose (Root Admin)

Provides emergency visibility, support, and debugging access in a system that
otherwise has no global accounts.

## 2. Event Owner (Event Admin Agent)

**Identity:** Possesses the event admin token shown at event creation, using the
format `/e/<event-id>/admin/<admin-token>`.

### Capabilities (Event Owner)

- Edit event title, description, date, time, and location.
- View all RSVPs for the event, including notes.
- Reveal RSVP magic links (hidden behind UI toggles).
- Toggle “Close RSVPs” to stop new submissions immediately or schedule an auto-close time that flips the toggle once the cutoff passes (times use the event’s timezone but are stored in UTC).
- Delete RSVP records on request (guests control edits).
- Delete the event early.
- Inspect decay status, scores, timestamps, and RSVP notes.

### Restrictions (Event Owner)

- Cannot modify RSVP data (names, pronouns, guest counts, notes, or status).
- May delete RSVP entries but cannot alter their content.
- Cannot access other events or the root admin dashboard.
- Cannot change decay scheduling.

### Purpose (Event Owner)

Lets event creators manage their own events without logins or extra accounts.

## 3. RSVP User Agent (Guest Identity)

**Identity:** Holds an RSVP token shown once after submission:
`/e/<event-id>/rsvp/<rsvp-token>`

### Capabilities (RSVP User)

- View and edit their RSVP (name, pronouns, status, guests, notes).
- Hide or unhide their RSVP from the public list; organizers still see it.
- Cancel or delete their RSVP.
- View event details.

### Restrictions (RSVP User)

- Cannot view or edit other RSVPs.
- Cannot access event admin or root admin capabilities.

### Purpose (RSVP User)

Provide a frictionless RSVP flow where each guest has one capability link per
event.

## 4. Public User Agent (Unauthenticated Visitor)

**Identity:** Anyone with `/e/<event-id>` or browsing the homepage.

### Capabilities (Public User)

- View public event details.
- Submit a new RSVP, receiving their token immediately.
- Browse public channels and events.

### Restrictions (Public User)

- Cannot edit existing RSVPs.
- Cannot see RSVP tokens or admin links.
- Cannot perform event admin or root admin actions.

### Purpose (Public User)

Enable open participation for communities such as FetLife groups or cigar
socials.

## Channel System

Channels are lightweight containers for events. They are not accounts.

### Lifecycle

- Created or reused when an event is created. Users supply a channel name and
  visibility (public or private).
- Names are slugified (lowercase, hyphenated). Matching slug + visibility reuses
  the channel and bumps `last_used_at`.
- Stored fields: `id`, `name`, `slug`, `visibility`, `score`, `created_at`,
  `last_used_at`.
- Channel scores decay hourly. Empty channels below the delete threshold are
  purged.

### Visibility Rules

- Public channels appear on the homepage filter and at `/channel/<slug>`. These
  listings only show public events.
- Private channels never appear in the UI but are reachable via `/channel/<slug>`
  if you know the slug. Listings show every event regardless of privacy.
- Event privacy is independent from channel privacy. A private event inside a
  public channel stays private.

### Guidance

- Encourage distinctive channel names (include cities, venues, or group names) to
  avoid slug collisions.
- Private channels rely on secrecy of the slug. Share them carefully.
- `/help` explains these rules for event creators and highlights the privacy
  model.

## 5. Internal Background Scheduler Agent

Runs inside the same container as the FastAPI app and executes APScheduler jobs.

### Capabilities (Scheduler)

- Run the decay cycle hourly.
- Apply exponential decay to event, RSVP, and channel scores.
- Hide stale or low score events from public listings.
- Delete expired events, RSVPs, and empty channels.
- Vacuum SQLite and update `last_accessed` timestamps.

### Restrictions (Scheduler)

- Cannot modify RSVP or event content outside the decay logic.
- Cannot alter the root admin token.
- Cannot create or edit events.

### Purpose (Scheduler)

Maintain overall system health and enforce ephemeral data lifecycles without an
external worker.

## Philosophy of OpenRSVP

1. **Accountless interaction** – all capabilities are magic links.
2. **Privacy first** – collect only name, pronouns, RSVP status, guest count,
   and notes. No email or analytics.
3. **Ephemeral data** – events and RSVPs decay naturally.
4. **Minimal admin power** – event owners cannot impersonate guests.
5. **Transparency** – guests trust that their RSVP cannot be edited silently.
6. **Simple deployment** – FastAPI, Jinja2, and APScheduler run in one
   container.

## Future Agents

- **Email Delivery Agent** – send RSVP confirmations and optional reminders.
- **Worker Container Agent** – offload decay tasks and long running processes.
- **Metrics Agent** (optional) – capture aggregated, non identifying stats.

## Documentation Targeting

- `/help` is written for end users (guests and event owners) inside the app.
- `/docs` is written for administrators/operators deploying or maintaining OpenRSVP.
