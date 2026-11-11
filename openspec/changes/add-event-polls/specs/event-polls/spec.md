# Spec Delta

## Purpose

Lets an event organizer attach short multiple-choice questions to an event, such as meal preferences or t-shirt sizes, and lets attendees answer them with their RSVP. Answers stay private to the organizer and the attendee who gave them.

## ADDED Requirements

### Requirement: Organizer defines poll questions
An organizer SHALL be able to attach poll questions to an event when creating it and when editing it. Each question MUST have a non-empty prompt of at most 200 characters, at least 2 and at most 12 options, and a setting that allows either one answer or several. Each option label MUST be non-empty and at most 64 characters, and labels MUST be unique within a question, ignoring case.

#### Scenario: Create event with a multi-select poll
- **WHEN** an organizer creates an event with the question "Meal preference" and the options "Vegan", "Vegetarian", and "Gluten-free", with several answers allowed
- **THEN** the event is created with that question and its options in the order given

#### Scenario: Create event without a poll
- **WHEN** an organizer creates an event without any poll questions
- **THEN** the event is created and has no poll questions

#### Scenario: Question with too few options is rejected
- **WHEN** an organizer submits a question with only one option
- **THEN** the request is rejected with a validation error and nothing is saved

#### Scenario: Duplicate option labels are rejected
- **WHEN** an organizer submits a question with the options "Vegan" and "vegan"
- **THEN** the request is rejected with a validation error and nothing is saved

### Requirement: Poll size limits
An event SHALL have at most 3 poll questions. A request that exceeds any poll limit MUST be rejected with a validation error rather than silently truncated.

#### Scenario: Fourth question is rejected
- **WHEN** an organizer submits 4 poll questions for an event
- **THEN** the request is rejected with a validation error and the event's existing poll is unchanged

#### Scenario: Thirteenth option is rejected
- **WHEN** an organizer submits a question with 13 options
- **THEN** the request is rejected with a validation error and the event's existing poll is unchanged

### Requirement: Organizer edits poll questions
An organizer SHALL be able to add, reword, reorder, and remove questions and options on an existing event. A renamed or reordered option MUST keep the answers already given for it. Removing an option or question MUST delete the answers given for it.

#### Scenario: Renaming an option keeps answers
- **GIVEN** an RSVP that selected the option "Vegitarian"
- **WHEN** the organizer renames that option to "Vegetarian"
- **THEN** the RSVP's answer shows "Vegetarian"

#### Scenario: Removing an option deletes its answers
- **GIVEN** an RSVP that selected the options "Vegan" and "Gluten-free"
- **WHEN** the organizer removes the option "Gluten-free"
- **THEN** the RSVP's answers contain only "Vegan"

#### Scenario: API update without poll field leaves poll unchanged
- **WHEN** an organizer updates an event through the API without including the poll questions field
- **THEN** the event's poll questions, options, and answers are unchanged

#### Scenario: API update with empty poll removes it
- **WHEN** an organizer updates an event through the API with an empty list of poll questions
- **THEN** all of the event's poll questions and their answers are removed

### Requirement: Attendee answers poll questions
An attendee SHALL be able to answer an event's poll questions when creating an RSVP and to change those answers later with their RSVP token. Every question is optional. A question that allows one answer MUST NOT accept more than one selected option.

#### Scenario: Answer a multi-select question
- **WHEN** an attendee RSVPs and selects "Vegan" and "Gluten-free" for a question that allows several answers
- **THEN** both options are saved as that RSVP's answers

#### Scenario: Skip a question
- **WHEN** an attendee RSVPs without selecting any option for a question
- **THEN** the RSVP is created and has no answers for that question

#### Scenario: Too many answers to a single-answer question
- **WHEN** an attendee selects two options for a question that allows one answer
- **THEN** the request is rejected with a validation error and no answers are changed

#### Scenario: Change answers later
- **GIVEN** an RSVP that selected "Vegan"
- **WHEN** the attendee uses their RSVP link or token to select "Vegetarian" instead
- **THEN** the RSVP's answers contain only "Vegetarian"

#### Scenario: Update RSVP without answers leaves them unchanged
- **WHEN** an attendee updates their RSVP through the API without including the answers field
- **THEN** their existing answers are unchanged

### Requirement: Answers must belong to the event
The system MUST reject any submitted answer that refers to an option that does not belong to the RSVP's event.

#### Scenario: Option from another event
- **WHEN** an attendee submits an answer containing an option ID from a different event
- **THEN** the request is rejected with a validation error and no answers are changed

#### Scenario: Unknown option
- **WHEN** an attendee submits an answer containing an option ID that does not exist
- **THEN** the request is rejected with a validation error and no answers are changed

### Requirement: Poll questions are public
Event pages and event API responses SHALL include the event's poll questions and options, in order, to anyone who can view the event, so that the RSVP form can be shown.

#### Scenario: Public event response includes questions
- **WHEN** an anonymous caller fetches an event that has a poll
- **THEN** the response includes each question's ID, prompt, whether it allows several answers, and its options with IDs and labels

### Requirement: Poll answers are private
Poll answers and totals MUST be visible only to the event organizer (admin token, or the instance root token) and, for a single RSVP's own answers, to the holder of that RSVP's token. Public event pages, public RSVP lists, and public API responses MUST NOT include answers or totals.

#### Scenario: Public event response excludes answers
- **WHEN** an anonymous caller fetches an event whose attendees have answered its poll
- **THEN** the response contains no answers and no totals for any RSVP, public or private

#### Scenario: Attendee sees only their own answers
- **WHEN** an attendee fetches their own RSVP with their RSVP token
- **THEN** the response includes that RSVP's answers and no other RSVP's answers

#### Scenario: Organizer sees every RSVP's answers
- **WHEN** the organizer lists the event's RSVPs with the admin token
- **THEN** each RSVP in the response includes its answers, including private RSVPs

### Requirement: Organizer sees poll totals
The organizer view SHALL show, for each option, how many RSVPs selected it and the total party size (attendee plus guests) of those RSVPs. Totals MUST count only approved RSVPs, with "yes" and "maybe" counted separately and "no" excluded.

#### Scenario: Totals count approved yes and maybe separately
- **GIVEN** three approved RSVPs selected "Vegan": one "yes" with 2 guests, one "yes" with no guests, and one "maybe" with no guests
- **WHEN** the organizer opens the event admin page
- **THEN** "Vegan" shows 2 RSVPs (4 people) attending and 1 RSVP (1 person) maybe

#### Scenario: Pending, rejected, and declining RSVPs are excluded
- **GIVEN** a pending RSVP, a rejected RSVP, and an approved "no" RSVP all selected "Vegan"
- **WHEN** the organizer opens the event admin page
- **THEN** "Vegan" shows no RSVPs in its totals

### Requirement: Poll data is deleted with its owner
Poll questions, options, and answers MUST be deleted when their event is deleted by any path, including organizer deletion, root admin deletion, API deletion, and the decay cycle. An RSVP's answers MUST be deleted when that RSVP is deleted.

#### Scenario: Event deleted by decay
- **GIVEN** an event with a poll and answered RSVPs
- **WHEN** the decay cycle deletes the event
- **THEN** none of that event's poll questions, options, or answers remain

#### Scenario: RSVP deleted
- **GIVEN** an RSVP with poll answers
- **WHEN** the attendee or the organizer deletes the RSVP
- **THEN** that RSVP's answers no longer exist and no longer count toward totals

### Requirement: Existing databases upgrade in place
Upgrading an existing database SHALL add poll support without changing existing events or RSVPs. Existing events MUST have no poll after the upgrade.

#### Scenario: Upgrade a database with existing events
- **GIVEN** a database at the previous schema version containing events and RSVPs
- **WHEN** the database upgrade runs
- **THEN** all existing events and RSVPs are unchanged, each event has no poll questions, and organizers can add polls to them
