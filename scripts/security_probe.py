from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, TypedDict


class HttpResult(TypedDict):
    status: int
    headers: dict[str, str]
    body: bytes


class _PreserveMethodRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        method = req.get_method()
        if method not in {"GET", "HEAD"} and code in {301, 302}:
            old = urllib.parse.urlparse(req.full_url)
            new = urllib.parse.urlparse(newurl)
            if old.scheme == "http" and new.scheme == "https" and old.netloc == new.netloc:
                return urllib.request.Request(
                    newurl,
                    data=req.data,
                    headers=dict(req.header_items()),
                    method=method,
                )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_URL_OPENER = urllib.request.build_opener(_PreserveMethodRedirectHandler())


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _rand_suffix(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _http_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json_body: object | None = None,
    timeout_s: float = 15.0,
) -> HttpResult:
    data = None if json_body is None else _json_bytes(json_body)
    request_headers = {
        "User-Agent": "OpenRSVP-SecurityProbe/1.0",
        "Accept": "application/json, text/plain, */*",
    }
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with _URL_OPENER.open(req, timeout=timeout_s) as resp:
            return {
                "status": int(resp.status),
                "headers": {k.lower(): v for k, v in resp.headers.items()},
                "body": resp.read(),
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": int(exc.code),
            "headers": {k.lower(): v for k, v in exc.headers.items()},
            "body": exc.read(),
        }


def _json(result: HttpResult) -> object:
    if not result["body"]:
        return None
    return json.loads(result["body"].decode("utf-8"))


@dataclass(frozen=True)
class CreatedEvent:
    event_id: str
    admin_token: str


@dataclass(frozen=True)
class CreatedRsvp:
    rsvp_token: str
    approval_status: str
    attendance_status: str
    is_private: bool


class Finding(TypedDict):
    severity: Literal["high", "medium", "low", "info"]
    title: str
    affected_endpoints: list[str]
    description: str
    repro_steps: list[str]
    evidence: dict[str, object]


def _create_event(
    *,
    base_url: str,
    title: str,
    is_private: bool,
    admin_approval_required: bool,
    max_attendees: int | None,
    channel_name: str | None,
    channel_visibility: str | None,
) -> CreatedEvent:
    start = _utcnow() + timedelta(days=7)
    payload: dict[str, object] = {
        "title": title,
        "description": "security probe",
        "start_time": _iso(start),
        "end_time": _iso(start + timedelta(hours=2)),
        "location": "Probe Location",
        "is_private": is_private,
        "admin_approval_required": admin_approval_required,
        "max_attendees": max_attendees,
        "rsvps_closed": False,
        "timezone_offset_minutes": 0,
    }
    if channel_name is not None:
        payload["channel_name"] = channel_name
    if channel_visibility is not None:
        payload["channel_visibility"] = channel_visibility

    resp = _http_request(
        method="POST",
        url=f"{base_url}/api/v1/events",
        json_body=payload,
    )
    if resp["status"] != 201:
        raise RuntimeError(f"Create event failed: {resp['status']} {resp['body'][:200]!r}")
    data = _json(resp)
    assert isinstance(data, dict)
    admin_token = str(data.get("admin_token") or "")
    event = data.get("event")
    assert isinstance(event, dict)
    event_id = str(event.get("id") or "")
    if not admin_token or not event_id:
        raise RuntimeError("Create event response missing admin_token or event.id")
    return CreatedEvent(event_id=event_id, admin_token=admin_token)


def _create_rsvp(
    *,
    base_url: str,
    event_id: str,
    name: str,
    attendance_status: str,
    guest_count: int,
    is_private_rsvp: bool,
) -> CreatedRsvp:
    payload: dict[str, object] = {
        "name": name,
        "attendance_status": attendance_status,
        "guest_count": guest_count,
        "is_private_rsvp": is_private_rsvp,
    }
    resp = _http_request(
        method="POST",
        url=f"{base_url}/api/v1/events/{event_id}/rsvps",
        json_body=payload,
    )
    if resp["status"] != 201:
        raise RuntimeError(f"Create RSVP failed: {resp['status']} {resp['body'][:200]!r}")
    data = _json(resp)
    assert isinstance(data, dict)
    rsvp = data.get("rsvp")
    assert isinstance(rsvp, dict)
    token = str(rsvp.get("rsvp_token") or "")
    approval_status = str(rsvp.get("approval_status") or "")
    attendance = str(rsvp.get("attendance_status") or "")
    private = bool(rsvp.get("is_private"))
    if not token:
        raise RuntimeError("Create RSVP response missing rsvp.rsvp_token")
    return CreatedRsvp(
        rsvp_token=token,
        approval_status=approval_status,
        attendance_status=attendance,
        is_private=private,
    )


def _get_public_event(*, base_url: str, event_id: str, bearer: str | None = None) -> dict:
    headers = None if bearer is None else {"Authorization": f"Bearer {bearer}"}
    resp = _http_request(
        method="GET",
        url=f"{base_url}/api/v1/events/{event_id}",
        headers=headers,
    )
    if resp["status"] != 200:
        raise RuntimeError(f"GET event failed: {resp['status']} {resp['body'][:200]!r}")
    data = _json(resp)
    assert isinstance(data, dict)
    event = data.get("event")
    assert isinstance(event, dict)
    return event


def _list_public_events(*, base_url: str, page: int, per_page: int) -> dict:
    query = urllib.parse.urlencode({"page": page, "per_page": per_page})
    resp = _http_request(method="GET", url=f"{base_url}/api/v1/events?{query}")
    if resp["status"] != 200:
        raise RuntimeError(f"GET /api/v1/events failed: {resp['status']} {resp['body'][:200]!r}")
    data = _json(resp)
    assert isinstance(data, dict)
    return data


def _assert_status(
    *,
    findings: list[Finding],
    title: str,
    expected: int,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    json_body: object | None = None,
    severity_on_mismatch: Literal["high", "medium", "low", "info"] = "high",
) -> HttpResult:
    resp = _http_request(method=method, url=url, headers=headers, json_body=json_body)
    if resp["status"] != expected:
        findings.append(
            {
                "severity": severity_on_mismatch,
                "title": title,
                "affected_endpoints": [f"{method} {urllib.parse.urlparse(url).path}"],
                "description": f"Expected HTTP {expected} but got {resp['status']}.",
                "repro_steps": [
                    f"Send `{method} {urllib.parse.urlparse(url).path}` without required auth.",
                ],
                "evidence": {
                    "expected_status": expected,
                    "actual_status": resp["status"],
                    "body_prefix": resp["body"][:200].decode("utf-8", errors="replace"),
                },
            }
        )
    return resp


def run_probe(*, base_url: str, scale_events: int, per_page_scan: int) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    created_events: list[CreatedEvent] = []
    meta: dict[str, object] = {
        "base_url": base_url,
        "started_at": _utcnow().isoformat(),
        "created_event_ids": [],
        "notes": [],
    }

    # 1) Private RSVP behavior (should hide details but keep aggregate counts).
    event_private_rsvp = _create_event(
        base_url=base_url,
        title=f"Probe Private RSVP {_rand_suffix()}",
        is_private=False,
        admin_approval_required=False,
        max_attendees=25,
        channel_name=None,
        channel_visibility=None,
    )
    created_events.append(event_private_rsvp)
    meta["created_event_ids"].append(event_private_rsvp.event_id)
    _create_rsvp(
        base_url=base_url,
        event_id=event_private_rsvp.event_id,
        name="Public Attendee",
        attendance_status="yes",
        guest_count=0,
        is_private_rsvp=False,
    )
    _create_rsvp(
        base_url=base_url,
        event_id=event_private_rsvp.event_id,
        name="Hidden Attendee",
        attendance_status="yes",
        guest_count=0,
        is_private_rsvp=True,
    )
    public_event_payload = _get_public_event(
        base_url=base_url, event_id=event_private_rsvp.event_id
    )
    rsvps = public_event_payload.get("rsvps")
    counts = public_event_payload.get("rsvp_counts")
    if not isinstance(rsvps, list) or not isinstance(counts, dict):
        findings.append(
            {
                "severity": "medium",
                "title": "Unexpected public event payload shape",
                "affected_endpoints": ["GET /api/v1/events/{event_id}"],
                "description": "Public event payload did not include expected RSVP list/counts structure.",
                "repro_steps": [
                    "Create an event and a mix of public/private RSVPs.",
                    "Fetch the event via `GET /api/v1/events/{event_id}`.",
                ],
                "evidence": {"event_id": event_private_rsvp.event_id},
            }
        )
    else:
        leaked_private = [r for r in rsvps if isinstance(r, dict) and r.get("is_private") is True]
        if leaked_private:
            findings.append(
                {
                    "severity": "high",
                    "title": "Private RSVP details exposed in public event API",
                    "affected_endpoints": ["GET /api/v1/events/{event_id}"],
                    "description": "At least one private RSVP appears in the public RSVP list.",
                    "repro_steps": [
                        "Create an event.",
                        "Create an RSVP with `is_private_rsvp=true`.",
                        "Call `GET /api/v1/events/{event_id}` and inspect `event.rsvps`.",
                    ],
                    "evidence": {
                        "event_id": event_private_rsvp.event_id,
                        "private_rsvp_objects_seen": len(leaked_private),
                    },
                }
            )
        private_count = counts.get("private")
        if private_count != 1:
            findings.append(
                {
                    "severity": "low",
                    "title": "Private RSVP aggregate count unexpected",
                    "affected_endpoints": ["GET /api/v1/events/{event_id}"],
                    "description": "Expected exactly one private RSVP count from the probe data.",
                    "repro_steps": [
                        "Create one private RSVP and one public RSVP.",
                        "Call `GET /api/v1/events/{event_id}` and inspect `event.rsvp_counts.private`.",
                    ],
                    "evidence": {
                        "event_id": event_private_rsvp.event_id,
                        "observed_private_count": private_count,
                    },
                }
            )

    # 2) Authorization checks on owner-only endpoints.
    _assert_status(
        findings=findings,
        title="Owner RSVP list accessible without auth",
        expected=401,
        method="GET",
        url=f"{base_url}/api/v1/events/{event_private_rsvp.event_id}/rsvps",
    )
    _assert_status(
        findings=findings,
        title="Owner RSVP list accessible with attendee token",
        expected=403,
        method="GET",
        url=f"{base_url}/api/v1/events/{event_private_rsvp.event_id}/rsvps",
        headers={"Authorization": "Bearer not-a-real-token"},
    )

    # 3) Private channel metadata leak via public event endpoints.
    channel_slug = f"probe-private-channel-{_rand_suffix()}"
    event_private_channel = _create_event(
        base_url=base_url,
        title=f"Probe Private Channel {_rand_suffix()}",
        is_private=False,
        admin_approval_required=False,
        max_attendees=25,
        channel_name=channel_slug,
        channel_visibility="private",
    )
    created_events.append(event_private_channel)
    meta["created_event_ids"].append(event_private_channel.event_id)
    event_payload = _get_public_event(base_url=base_url, event_id=event_private_channel.event_id)
    channel = event_payload.get("channel")
    if isinstance(channel, dict) and channel.get("visibility") == "private":
        findings.append(
            {
                "severity": "medium",
                "title": "Private channel metadata exposed in public event API",
                "affected_endpoints": ["GET /api/v1/events/{event_id}", "GET /api/v1/events"],
                "description": (
                    "A public event in a private channel returns the channel object (name/slug/visibility) "
                    "via the public API, which can deanonymize private channels."
                ),
                "repro_steps": [
                    "Create an event with `channel_visibility=private` and `is_private=false`.",
                    "Call `GET /api/v1/events/{event_id}` and check if `event.channel` is present.",
                ],
                "evidence": {
                    "event_id": event_private_channel.event_id,
                    "channel_keys": sorted(channel.keys()),
                },
            }
        )

    # 4) Approval-required events: check for public leakage of unapproved RSVP presence via yes_count.
    event_approval = _create_event(
        base_url=base_url,
        title=f"Probe Approval {_rand_suffix()}",
        is_private=False,
        admin_approval_required=True,
        max_attendees=25,
        channel_name=None,
        channel_visibility=None,
    )
    created_events.append(event_approval)
    meta["created_event_ids"].append(event_approval.event_id)
    pending_rsvp = _create_rsvp(
        base_url=base_url,
        event_id=event_approval.event_id,
        name="Pending Attendee",
        attendance_status="yes",
        guest_count=0,
        is_private_rsvp=False,
    )
    if pending_rsvp.approval_status != "pending":
        meta["notes"].append(
            f"Expected pending RSVP for approval event, got {pending_rsvp.approval_status!r}"
        )
    approval_payload = _get_public_event(base_url=base_url, event_id=event_approval.event_id)
    yes_count = approval_payload.get("yes_count")
    public_counts = (
        approval_payload.get("rsvp_counts", {}).get("public")
        if isinstance(approval_payload.get("rsvp_counts"), dict)
        else None
    )
    if yes_count not in (0, None) and public_counts == 0:
        findings.append(
            {
                "severity": "low",
                "title": "Approval-required event leaks attendance via yes_count",
                "affected_endpoints": ["GET /api/v1/events/{event_id}", "GET /api/v1/events"],
                "description": (
                    "When admin approval is required, a pending 'yes' RSVP can increment `yes_count` even "
                    "though no RSVPs are publicly visible yet. This may leak the presence of unapproved attendees."
                ),
                "repro_steps": [
                    "Create an event with `admin_approval_required=true`.",
                    "Create an RSVP with `attendance_status=yes` (it will be pending).",
                    "Call `GET /api/v1/events/{event_id}` and compare `yes_count` vs `rsvp_counts.public`.",
                ],
                "evidence": {
                    "event_id": event_approval.event_id,
                    "yes_count": yes_count,
                    "public_rsvp_count": public_counts,
                },
            }
        )

    # 5) Create additional events to stress basic flows (kept small and throttled).
    for _ in range(max(0, scale_events - 3)):
        ev = _create_event(
            base_url=base_url,
            title=f"Probe Scale {_rand_suffix()}",
            is_private=False,
            admin_approval_required=False,
            max_attendees=50,
            channel_name=None,
            channel_visibility=None,
        )
        created_events.append(ev)
        meta["created_event_ids"].append(ev.event_id)
        for idx in range(3):
            _create_rsvp(
                base_url=base_url,
                event_id=ev.event_id,
                name=f"Scale Guest {idx}",
                attendance_status="yes",
                guest_count=0,
                is_private_rsvp=(idx == 2),
            )
        time.sleep(0.05)

    # 6) Lightweight public scan for suspicious fields in RSVP objects.
    scan_summary: dict[str, object] = {
        "events_scanned": 0,
        "events_with_private_channel_in_payload": 0,
        "rsvp_objects_with_token_field": 0,
        "rsvp_objects_with_internal_id_field": 0,
    }
    page = 1
    while page <= 5:
        listing = _list_public_events(base_url=base_url, page=page, per_page=per_page_scan)
        events = listing.get("events")
        if not isinstance(events, list) or not events:
            break
        for evt in events:
            if not isinstance(evt, dict):
                continue
            scan_summary["events_scanned"] = int(scan_summary["events_scanned"]) + 1
            channel = evt.get("channel")
            if isinstance(channel, dict) and channel.get("visibility") == "private":
                scan_summary["events_with_private_channel_in_payload"] = (
                    int(scan_summary["events_with_private_channel_in_payload"]) + 1
                )
            for rsvp in evt.get("rsvps") or []:
                if not isinstance(rsvp, dict):
                    continue
                if "rsvp_token" in rsvp:
                    scan_summary["rsvp_objects_with_token_field"] = (
                        int(scan_summary["rsvp_objects_with_token_field"]) + 1
                    )
                if "id" in rsvp:
                    scan_summary["rsvp_objects_with_internal_id_field"] = (
                        int(scan_summary["rsvp_objects_with_internal_id_field"]) + 1
                    )
        page += 1

    meta["public_scan_summary"] = scan_summary
    meta["finished_at"] = _utcnow().isoformat()
    meta["_created_events"] = created_events
    return findings, meta


def _render_report(*, findings: list[Finding], meta: dict) -> str:
    lines: list[str] = []
    lines.append(f"# OpenRSVP Dev Security Probe Report ({_utcnow().date().isoformat()})")
    lines.append("")
    lines.append(f"- Target: `{meta.get('base_url')}`")
    lines.append(f"- Started: `{meta.get('started_at')}`")
    lines.append(f"- Finished: `{meta.get('finished_at')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    sev_order = ["high", "medium", "low", "info"]
    counts = {sev: 0 for sev in sev_order}
    for finding in findings:
        counts[finding["severity"]] += 1
    lines.append(
        "- Findings: "
        + ", ".join(f"{sev}={counts[sev]}" for sev in sev_order)
    )
    scan = meta.get("public_scan_summary") or {}
    if isinstance(scan, dict):
        lines.append(f"- Public events scanned: `{scan.get('events_scanned')}`")
        lines.append(
            f"- Events exposing private channel metadata in listings: `{scan.get('events_with_private_channel_in_payload')}`"
        )
        lines.append(
            f"- RSVP objects leaking `rsvp_token`: `{scan.get('rsvp_objects_with_token_field')}`"
        )
        lines.append(
            f"- RSVP objects leaking internal `id`: `{scan.get('rsvp_objects_with_internal_id_field')}`"
        )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not findings:
        lines.append("- No issues detected by this probe.")
        return "\n".join(lines) + "\n"

    for idx, finding in enumerate(findings, start=1):
        lines.append(f"### {idx}. [{finding['severity'].upper()}] {finding['title']}")
        lines.append("")
        lines.append("**Affected endpoints**")
        for ep in finding["affected_endpoints"]:
            lines.append(f"- `{ep}`")
        lines.append("")
        lines.append("**Description**")
        lines.append(f"- {finding['description']}")
        lines.append("")
        lines.append("**Repro (tokens redacted)**")
        for step in finding["repro_steps"]:
            lines.append(f"- {step}")
        lines.append("")
        lines.append("**Evidence (sanitized)**")
        for key, value in finding["evidence"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Authorized OpenRSVP dev security probe")
    parser.add_argument(
        "--base-url",
        required=True,
        help="Base URL for the target OpenRSVP instance",
    )
    parser.add_argument(
        "--scale-events",
        type=int,
        default=8,
        help="How many probe events to create (default: %(default)s)",
    )
    parser.add_argument(
        "--per-page-scan",
        type=int,
        default=50,
        help="per_page used when scanning public events (default: %(default)s)",
    )
    parser.add_argument(
        "--out-json",
        default="security_probe_results.json",
        help="Write raw results JSON to this path (default: %(default)s)",
    )
    parser.add_argument(
        "--out-report",
        default="docs/security_probe_report.md",
        help="Write markdown report to this path (default: %(default)s)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete probe-created events at the end (best-effort)",
    )
    args = parser.parse_args(argv)

    findings, meta = run_probe(
        base_url=args.base_url.rstrip("/"),
        scale_events=max(3, args.scale_events),
        per_page_scan=max(1, min(args.per_page_scan, 50)),
    )

    created_events = meta.pop("_created_events", [])
    if args.cleanup and isinstance(created_events, list):
        failures: list[str] = []
        for item in created_events:
            if not isinstance(item, CreatedEvent):
                continue
            resp = _http_request(
                method="DELETE",
                url=f"{args.base_url.rstrip('/')}/api/v1/events/{item.event_id}",
                headers={"Authorization": f"Bearer {item.admin_token}"},
            )
            if resp["status"] != 204:
                failures.append(f"{item.event_id}:{resp['status']}")
        if failures:
            notes = meta.get("notes")
            if not isinstance(notes, list):
                notes = []
                meta["notes"] = notes
            notes.append(f"Cleanup failures: {len(failures)} events")

    out_json = {
        "meta": meta,
        "findings": findings,
    }
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(out_json, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    report = _render_report(findings=findings, meta=meta)
    with open(args.out_report, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Wrote `{args.out_json}` and `{args.out_report}`")
    print(f"Findings: {len(findings)} (high/medium/low/info)")
    for finding in findings:
        print(f"- {finding['severity']}: {finding['title']}")

    return 0 if not any(f["severity"] in {"high", "medium"} for f in findings) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
