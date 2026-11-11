# Design

## Context

See `proposal.md` for motivation and `specs/event-polls/spec.md` for the required behavior.

Relevant facts about the current code:

- Models are plain SQLAlchemy `declarative_base` classes in `openrsvp/models.py`, not SQLModel, despite what `AGENTS.md` says. This design follows the code.
- `Event` owns `RSVP` and `Message` rows through ORM relationships with `cascade="all, delete-orphan"` / `"all, delete"`. Every deletion path (`api.py` organizer, root, and API deletes; `decay.py:155,190`) calls `session.delete(...)` on the ORM object.
- SQLite foreign key enforcement (`PRAGMA foreign_keys=ON`) is not turned on, so `ondelete="CASCADE"` in the schema does nothing at runtime. **ORM cascades are what actually delete child rows.**
- HTML handlers are synchronous `def` functions that take explicit `Form(...)` parameters. They cannot `await request.form()` to read dynamic field names.
- `_serialize_rsvp` is shared by public, attendee, and organizer responses. It already uses opt-in flags (`include_token`, `include_internal_id`, `include_messages`) to add private fields.
- Headcount rules live in `_rsvp_stats` (approved only; `yes`, `maybe`, and `no` counted separately). `RSVP.party_size` (1 + `guest_count`) is the shared definition of party size.
- Guests per RSVP are limited per event by `Event.max_guests` (0–5, default 5; read through `Event.guest_limit`). Capacity uses `RSVP.holds_spot`, which counts "yes" RSVPs that are pending or approved, but not rejected.
- RSVP edits through the RSVP link are allowed even after RSVPs close (`save_rsvp` does not check `rsvps_closed`).
- Repeating events: `crud.create_event_series` creates 2–52 independent `Event` rows that share a `series_id`. The series admin page only lists occurrences and links to each one's own admin page. There is no series-wide editing, and the API has no series creation.
- `duplicate_event` (`POST /e/{id}/admin/{token}/duplicate`) creates a new event from an existing event's settings, without its RSVPs.
- The migration chain ends at `0004_event_max_guests`.

## Goals / Non-Goals

**Goals:**
- Fit the existing ownership chain (Event → RSVP → child rows) so cleanup by decay and deletion needs no new code.
- Keep the change additive: new tables only, and no changes to existing columns or existing response fields.
- Use the same validation rules for the HTML forms and the JSON API by putting them in CRUD helpers.

**Non-Goals:**
- Free-text answers. RSVP notes already cover these.
- Required questions.
- Per-guest answers or quantities. Answers describe the whole party; see Open Questions.
- Public results or per-question visibility settings.
- Reusable poll templates across events or channels.
- Exporting answers (CSV).
- Editing one poll across a whole series. Each occurrence's poll is edited on its own, like every other occurrence setting.

## Decisions

### D1. Three normalized tables instead of JSON columns

```
poll_questions (id PK, event_id FK→events, prompt, allow_multiple, position)
poll_options   (id PK, question_id FK→poll_questions, label, position)
poll_answers   (rsvp_id FK→rsvps, option_id FK→poll_options, PK(rsvp_id, option_id))
```

- `Event.poll_questions` (ordered by `position`) and `PollQuestion.options` (ordered by `position`) use `cascade="all, delete-orphan"`.
- `RSVP.poll_answers` and `PollOption.answers` use `cascade="all, delete-orphan"`. An answer therefore disappears when either its RSVP or its option is deleted.
- Each selected option is one `poll_answers` row. The composite primary key prevents duplicates, and totals are a simple `GROUP BY option_id`.

**Alternative considered:** a JSON column on `events` (question definitions) and on `rsvps` (selected labels). It needs no new tables, but totals would have to be computed in Python, renaming an option would break existing answers, and keeping options and answers consistent would rely on application code alone. Rejected.

### D2. Answers are per RSVP and represent the whole party

An answer means "this applies to my party." Totals report both the number of RSVPs that selected an option and the total party size of those RSVPs, so organizers can estimate the upper bound ("Vegan: 2 RSVPs, up to 4 people").

Party size comes from `RSVP.party_size`, so totals automatically respect each event's `max_guests`. With `max_guests = 0`, every party is 1 and the two numbers are equal.

**Alternative considered:** a `quantity` column on `poll_answers`, capped at the RSVP's `party_size`. This is more accurate for per-person questions like t-shirt sizes, but it adds a number input to every option and needs extra validation whenever `guest_count` or the event's `max_guests` changes. Deferred; see Open Questions.

### D3. Option identity when editing: ID first, then label

`set_event_poll(session, event, questions)` replaces an event's whole poll in one call:

1. A submitted question or option that carries an `id` belonging to this event is updated in place.
2. A submitted option without an `id` is matched to an existing option of the same question by label, ignoring case. This lets the HTML textarea editor keep answers when only the order or capitalization changes.
3. Anything left unmatched is created. Existing questions or options that were not submitted are deleted, which also deletes their answers through the cascade.

**Rename handling:** The API keeps answers on a rename by sending the option `id`. The HTML editor sends hidden `option_id` fields alongside each label (see D5) for the same result.

**Alternative considered:** treating options as immutable and creating a new option on every edit. That loses answers whenever someone fixes a typo. Rejected.

### D4. Validation lives in CRUD and raises `ValueError`

`set_event_poll` and `set_rsvp_answers(session, rsvp, option_ids)` check every rule in the spec: limits, unique labels, options belonging to the event, and at most one answer for single-answer questions. They raise `ValueError` before changing anything. API handlers turn this into `HTTPException(400)`, and HTML handlers re-render the form with an error message, the same pattern `ensure_channel` errors already use (`_channel_error_message`).

Limits are module constants in `crud.py` (`MAX_POLL_QUESTIONS = 3`, `MAX_POLL_OPTIONS = 12`, `MAX_POLL_PROMPT_LENGTH = 200`, `MAX_POLL_LABEL_LENGTH = 64`). They are not settings in `config.py`, because changing them would change the API contract.

Out-of-range values are rejected rather than clamped like `guest_count`, because silently dropping a poll option would lose data the organizer typed (see the spec's "Poll size limits" requirement).

### D5. HTML form encoding without dynamic field names

- **RSVP forms:** every checkbox or radio uses `name="poll_option"` with `value="{{ option.id }}"`. Option IDs are globally unique, so the handler declares `poll_option: list[str] = Form([])` and passes the list to `set_rsvp_answers`. The handlers can stay synchronous `def`.
- **Organizer editor:** three fixed question slots (`MAX_POLL_QUESTIONS`), numbered `poll_0_*` to `poll_2_*`, which gives a fixed set of `Form` parameters. Each slot has:
  - `poll_N_id` (hidden, empty for a new question)
  - `poll_N_prompt` (an empty prompt means the slot is unused)
  - `poll_N_multiple` (checkbox)
  - Repeated `poll_N_option_id` and `poll_N_option_label` pairs, one row per existing option. Clearing a label removes that option.
  - `poll_N_new_options`, a textarea with one new option per line.

  A Python helper, `_parse_poll_form(...)`, combines these into the same question structure the API payload uses, so existing options keep their IDs and answers when renamed (D3, step 1). On the create page there are no existing options, so only the textarea is shown. Jinja only loops over slots and options; all parsing happens in Python.

### D6. Privacy enforcement in the serializers

- `_serialize_event` gets a `poll_questions` key containing only IDs, prompts, `allow_multiple`, and option IDs and labels. It contains no counts and is always included.
- `_serialize_rsvp` gets `include_answers: bool = False`. It is set only in `api_get_own_rsvp`, `api_update_own_rsvp`, `api_create_rsvp` (the creator's own response), and `api_list_event_rsvps` (admin).
- `_poll_tallies(event)` sits next to `_rsvp_stats` and uses the same rule: approved RSVPs only, with "yes" and "maybe" counted separately. It is called only from `event_admin` and `api_list_event_rsvps` (added as `stats["poll"]`). This deliberately differs from capacity (`holds_spot`), which also counts pending "yes" RSVPs. Totals answer "what do confirmed attendees need," so they match the confirmed headcount the organizer already sees.
- The HTML organizer view receives tallies and per-RSVP answers in its context. Public templates (`event.html`, channel pages) receive no answer data at all, so a template mistake cannot leak it.

### D7. The poll is edited with the event; answers are edited with the RSVP

The event APIs take a `questions` field on create and on update. For update, the field is optional with three states: omitted means unchanged, `[]` means remove all, and a list means replace. The RSVP APIs take an `answers` field (option IDs) that follows the same three states. No separate poll endpoints are added, because the existing resources already have the right owner, the right auth checks, and the right lifecycle.

### D8. Series occurrences and duplicates get their own copy of the poll

A new `copy_event_poll(session, source, target)` CRUD helper creates new question and option rows on `target` with the same prompts, labels, `allow_multiple` settings, and positions. It never copies answers.

- **Series:** `create_event_series` takes the same `questions` structure as `set_event_poll`. It validates the structure once, applies it to the first occurrence, and copies that poll to the rest, so a validation error creates no events at all.
- **Duplicate:** `duplicate_event` calls `copy_event_poll(event, new_event)` after `create_event`.

**Alternative considered:** one shared poll per series, keyed by `series_id`. Answers would then mix across occurrences. Every other occurrence setting is already copied and edited per event, and the series admin page has no editing. Rejected.

## Risks / Trade-offs

- **[Risk] Small totals could identify someone even though they are organizer-only.** → Acceptable, because the organizer already sees every RSVP's name. Totals are never public (spec: "Poll answers are private").
- **[Risk] A new public serializer field could accidentally include answers.** → Answers are opt-in through `include_answers`, and the tests include leak checks on the public event and public RSVP list responses, covering private RSVPs.
- **[Risk] The organizer edits the poll after people have answered.** → Option identity rules (D3) keep answers on rename and reorder. Removals delete answers, which the spec requires and the admin UI should warn about.
- **[Risk] Changing a question from multiple answers to one answer when RSVPs already have several.** → Existing answers are kept and still counted. The one-answer rule applies the next time that attendee saves. Documented here rather than blocking the organizer's edit.
- **[Trade-off] Per-party answers are less accurate for per-person questions** (D2). → Totals show RSVP count and party size together, and `quantity` is the planned follow-up.
- **[Trade-off] Pending "yes" RSVPs hold capacity but are not in poll totals** (D6). → On approval-required events, the admin page should label totals as "approved RSVPs" so the gap is visible.
- **[Trade-off] A poll fix has to be repeated on each series occurrence** (D8). → This matches how every other occurrence setting works today. If organizers complain, series-wide editing is a separate change.
- **[Risk] ORM cascades are the only cleanup**, because SQLite FK enforcement is off. → Never delete events or RSVPs with bulk `delete()` statements. A test deletes an event through the decay cycle and checks that the poll tables are empty.

## Migration Plan

1. Add `openrsvp/alembic/versions/0005_event_polls.py` (`down_revision = "0004_event_max_guests"`). It creates the three tables with foreign keys (`ondelete="CASCADE"`, for documentation and future FK enforcement) and indexes on `poll_questions.event_id` and `poll_options.question_id`. Inline `ForeignKey`s in `op.create_table` are fine in SQLite. The constraint-naming problem that `0003_repeating_events` avoided only affects `batch_alter_table` on existing tables, and this migration changes no existing table.
2. `downgrade()` drops the three tables in reverse order.
3. Deploy as usual. `upgrade-db` (via `storage.upgrade_database`) backs up the SQLite file before upgrading.
4. Rollback: run the downgrade, then deploy the previous release. This loses only poll data.

## Open Questions

- Should per-person quantities (D2 alternative) be added once organizers use t-shirt-size polls? This can be added later as a column with a default of 1, without changing this design's behavior.
- Should organizers be able to make a question's totals public, for example "12 people want pizza"? This would be a new opt-in requirement and would not change the default described here.
