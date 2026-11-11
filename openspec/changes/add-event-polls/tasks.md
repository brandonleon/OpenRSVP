# Tasks

## 1. Data model and migration

- [ ] 1.1 Add `PollQuestion`, `PollOption`, and `PollAnswer` to `openrsvp/models.py`, plus `Event.poll_questions` and `RSVP.poll_answers` relationships with `cascade="all, delete-orphan"` and `position` ordering (design D1). Verify with `uv run pyright` and a test in `tests/test_crud.py` that `session.delete(event)` leaves all three tables empty.
- [ ] 1.2 Add `openrsvp/alembic/versions/0005_event_polls.py` (`down_revision = "0004_event_max_guests"`) that creates the three tables with foreign keys and indexes on `event_id` and `question_id`, and drops them in `downgrade()`. Verify with a `tests/test_migrations.py` case, modeled on the existing pre-0004 test, that upgrades a database at `0004_event_max_guests` containing events, RSVPs, and a series to head, checks that the existing rows are unchanged and the new tables exist, and checks that `_alembic_head()` is `0005_event_polls`.

## 2. CRUD helpers and validation

- [ ] 2.1 Add poll limit constants and `set_event_poll(session, event, questions)` to `openrsvp/crud.py`. It matches questions and options by ID, then by label ignoring case, creates what is new, deletes what was left out, and raises `ValueError` before changing anything when a rule is broken (design D3, D4). Verify with `tests/test_crud.py` cases for create, reorder, rename by ID, label match ignoring case, removal deleting answers, too few or too many options, more than 3 questions, duplicate labels, and over-length prompts and labels.
- [ ] 2.2 Add `set_rsvp_answers(session, rsvp, option_ids)` to `openrsvp/crud.py`. It replaces the RSVP's answers and rejects option IDs that belong to another event or do not exist, and more than one option for a single-answer question, with `ValueError` and no partial writes. Verify with `tests/test_crud.py` cases for multi-select, skipping a question, an option from another event, an unknown ID, and too many answers to a single-answer question.
- [ ] 2.3 Add `_poll_tallies(event)` next to `_rsvp_stats` in `openrsvp/api.py`. For each option it returns the yes and maybe RSVP counts and party sizes (`RSVP.party_size`), using approved RSVPs only (design D6). Verify with a test reproducing both "Organizer sees poll totals" scenarios from the spec, plus a case on an event with `max_guests = 0` where party size equals RSVP count.
- [ ] 2.4 Add `copy_event_poll(session, source, target)` to `openrsvp/crud.py` (design D8). It copies questions and options with their order and settings, and never answers. Verify with a `tests/test_crud.py` case that the copy has new IDs, matching prompts, labels, and order, no answers, and that editing the copy leaves the source unchanged.

## 3. JSON API

- [ ] 3.1 Add `PollQuestionPayload` and `PollOptionPayload` and an optional `questions` field to `EventCreatePayload` and `EventUpdatePayload`. Wire them into `api_create_event` and `api_update_event` (omitted means unchanged, `[]` means remove all), converting `ValueError` to `HTTPException(400)`. Verify with `tests/test_api.py` cases for create with a poll, update that renames by ID while keeping answers, update without the field, update with `[]`, and a 400 on limit violations.
- [ ] 3.2 Add `poll_questions` (IDs, prompts, `allow_multiple`, and options only) to `_serialize_event`, and `include_answers` to `_serialize_rsvp` (design D6). Verify with a test that `GET /api/v1/events/{id}` returns the questions and options and contains no answers or totals, including when private RSVPs have answered.
- [ ] 3.3 Add an optional `answers` field (option IDs) to `RSVPCreatePayload` and `RSVPUpdatePayload`. Wire it into `api_create_rsvp` and `api_update_own_rsvp`, and return answers from those endpoints and `api_get_own_rsvp`. Verify with tests for create with answers, update self (changing answers, and omitting the field), getting your own RSVP (only your own answers), and a 400 for an option from another event.
- [ ] 3.4 Include answers for every RSVP and `stats["poll"]` totals in `api_list_event_rsvps`. Verify with a test that the admin token sees answers for public and private RSVPs and that the totals match the spec scenarios.
- [ ] 3.5 Document the `questions` and `answers` fields, `poll_questions` in event responses, and `stats.poll` in `docs/api.html`, and state that answers are organizer-only and self-only. Verify by checking that the documented request examples match the field names in the payload models.

## 4. Organizer HTML UI

- [ ] 4.1 Add `_parse_poll_form(...)` and a `Depends`-based poll form dependency that declares the fixed `poll_N_*` fields once (design D5). Verify with unit tests that turn sample form data into the same structure the API uses, including empty slots, renamed existing options, and new options from the textarea.
- [ ] 4.2 Add the poll editor to `event_create.html` (prompt, multiple-answers checkbox, and options textarea for each of 3 slots) and wire it into the single-event path of `submit_event`. Validation errors re-render the form with a message. Verify with a `tests/test_api.py` form POST test that creates an event with a poll, and one that gets an error response for a one-option question.
- [ ] 4.3 Add the poll editor to the `event_admin.html` settings form (existing options as one row each with a hidden ID, plus a new-options textarea, and a warning that removing an option deletes its answers) and wire it into `save_event_admin`. Verify with a form POST test that renames an option and checks that its answers are kept.
- [ ] 4.4 Add a totals table to `event_admin.html` (labeled as counting approved RSVPs when the event requires approval) and answer chips to `partials/admin_rsvp_card.html`, passing tallies and answers from `event_admin` and from `_admin_rsvp_partial_response`, which re-renders the card. Verify with a test that the admin page HTML contains the totals and an RSVP's chosen labels.
- [ ] 4.5 Copy the poll to series occurrences and duplicates (design D8). Pass the parsed questions into `create_event_series` from the series path of `submit_event`, validating before any event is created, and call `copy_event_poll` in `duplicate_event`. Verify with `tests/test_api.py` form POST tests for all three "Poll is copied to series occurrences and duplicates" scenarios, plus one where an invalid poll on a series request creates no events.

## 5. Attendee HTML UI

- [ ] 5.1 Render poll questions on `rsvp_form.html` (checkboxes, or radios for single-answer questions, all `name="poll_option"`) and accept `poll_option: list[str] = Form([])` in `create_rsvp_view`. Invalid answers re-render the form with an error. Verify with form POST tests for creating an RSVP with answers and for rejecting an option from another event.
- [ ] 5.2 Render the same inputs, pre-checked with the RSVP's answers, on `rsvp_edit.html` and accept them in `save_rsvp`. Verify with a form POST test that changes answers through the RSVP link.
- [ ] 5.3 Confirm that public templates (`event.html`, channel pages) receive no answer data. Verify with a test that fetches `/e/{id}` for an event with answered RSVPs and asserts that no option label from an answer appears in the page's RSVP list area and that no totals are rendered.

- [ ] 5.4 Update the in-app help pages: `help/features.html` (what polls are), `help/rsvp.html` (answering and changing answers), and `help/privacy.html` (only the organizer and you can see your answers). Verify by loading `/help/features`, `/help/rsvp`, and `/help/privacy` and checking that the new sections render and match the spec's privacy rules.

## 6. Integration and release

- [ ] 6.1 Add a decay test that creates an event with a poll and answers, makes it eligible for deletion, runs `run_decay_cycle()`, and checks that all three poll tables are empty.
- [ ] 6.2 Add a `CHANGELOG.md` entry describing event polls and their privacy model. Verify that the entry appears under `## [Unreleased]`.
- [ ] 6.3 Run `uv run ruff check`, `uv run ruff format --check`, `uv run pyright`, and `uv run pytest`, and verify all pass.
- [ ] 6.4 Run the app manually (`uv run python main.py`): create an event with a meal-preference poll, RSVP twice with different answers, and check the admin totals and that the public event page shows no answers. Then create a weekly series of 2 with a poll and duplicate one event, and check that each new event has its own poll with no answers.

## Workflow follow-up

- Archive the change with `/opsx:archive` once it is merged, which creates `openspec/specs/event-polls/spec.md` as the first main spec.
