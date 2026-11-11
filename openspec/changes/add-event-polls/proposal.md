# Proposal

## Why

Organizers often need a little structured information from attendees, such as meal preferences (Vegan, Vegetarian, Gluten-free) or t-shirt sizes. Today the only way to collect it is a free-text RSVP note, which organizers then have to count by hand. A small poll attached to the event lets attendees pick from fixed options and gives organizers automatic totals.

## What Changes

- Organizers can attach up to 3 short poll questions to an event, each with up to 12 fixed options. A question allows either one answer or several.
- Polls can be set when creating the event and edited later from the organizer page, with matching JSON API support (`EventCreatePayload`, `EventUpdatePayload`).
- Attendees answer the questions on the RSVP form and can change their answers later through their RSVP link or the `/rsvps/self` API. All questions are optional.
- Organizers see per-option totals and each RSVP's answers on the event admin page and in the admin RSVP list API.
- Poll answers are private to the organizer and the attendee who gave them. Public event pages and public API responses include the questions and options, so the RSVP form can render, but never include answers or totals.
- Poll data belongs to its event and RSVP, so deleting an event or RSVP (including decay deletion) removes it automatically.
- A new Alembic migration adds the poll tables to existing databases without changing existing tables.

No breaking changes. Existing events have no poll, and existing API fields are unchanged.

## Capabilities

### New Capabilities
- `event-polls`: Organizer-defined multiple-choice questions attached to an event, attendee answers attached to their RSVP, who can see answers and totals, and how both are cleaned up.

### Modified Capabilities
<!-- None: there are no existing specs under openspec/specs/ yet. -->

## Impact

- **Data model** (`openrsvp/models.py`): new `PollQuestion`, `PollOption`, and `PollAnswer` models. `Event` gains a `poll_questions` relationship and `RSVP` gains a `poll_answers` relationship, both with cascade delete.
- **Migration**: new `openrsvp/alembic/versions/0003_event_polls.py`.
- **CRUD** (`openrsvp/crud.py`): new `set_event_poll` and `set_rsvp_answers` helpers.
- **HTML routes** (`openrsvp/api.py`): `submit_event`, `save_event_admin`, `create_rsvp_view`, `save_rsvp`, `event_admin`, `edit_rsvp`.
- **JSON API** (`openrsvp/api.py`): event create, update, and get; RSVP create, get self, and update self; admin RSVP list. Answers are added through a new opt-in `include_answers` flag on `_serialize_rsvp`, so public responses are unchanged.
- **Templates**: `event_create.html`, `event_admin.html`, `rsvp_form.html`, `rsvp_edit.html`, `partials/admin_rsvp_card.html`.
- **Docs**: `docs/api.html`, `CHANGELOG.md`.
- **Tests**: `tests/test_api.py`, `tests/test_crud.py`, `tests/test_migrations.py`.
- No new dependencies.
