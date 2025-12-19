# OpenRSVP Dev Security Findings

This repo includes `scripts/security_probe.py`, a small authorized probe you can run against a dev deployment to check for common privacy/auth leaks.

## How To Re-Run The Probe

Run against your dev host:

`python3 scripts/security_probe.py --base-url https://openrsvp.smokehouse.kaotic.cc --scale-events 12`

To delete the probe-created events afterwards:

`python3 scripts/security_probe.py --base-url https://openrsvp.smokehouse.kaotic.cc --scale-events 12 --cleanup`

Outputs:

- `docs/security_probe_report.md` (markdown summary; tokens redacted)
- `security_probe_results.json` (raw JSON; tokens redacted)

## Findings (Observed On Dev Host)

### 1) [MEDIUM] Private channel metadata exposed via public event APIs

**Impact**

Public API responses can include a private channel’s `name`/`slug`/`visibility` for events in that channel. This defeats the “hide the private channel badge” behavior the HTML UI already implements and can deanonymize private channels when a public event is placed inside them.

**Affected**

- `GET /api/v1/events`
- `GET /api/v1/events/{event_id}`

**Repro (tokens redacted)**

1. Create an event with `channel_visibility=private` and `is_private=false`.
2. Call `GET /api/v1/events/{event_id}` as an unauthenticated user.
3. Observe that `event.channel` is present even though the channel is private.

**Fix (implemented in this patch)**

- `openrsvp/api.py` now omits `event.channel` when the channel is private unless the caller is event admin or root admin.
- Regression test added: `tests/test_api.py` (`test_public_event_in_private_channel_hides_channel_in_public_api`).

### 2) [LOW] Approval-required events leak “pending yes” attendance via `yes_count`

**Impact**

For `admin_approval_required=true` events, a pending “yes” RSVP could increment `yes_count` even when `rsvp_counts.public == 0`, revealing unapproved attendance.

**Affected**

- `GET /api/v1/events`
- `GET /api/v1/events/{event_id}`
- `GET /api/v1/channels/{slug}` (event list serialization)

**Fix (implemented in this patch)**

- For approval-required events, public API responses now compute `yes_count` from **approved** `yes` RSVPs only (unless caller is event admin/root).
- Regression coverage added by extending `tests/test_api.py` (`test_pending_rsvp_hidden_until_approved`).

## Notes

- The dev host will continue to show the findings until you deploy the updated code.
- The probe is intentionally read/write (it creates a handful of events/RSVPs); don’t run it against production.
