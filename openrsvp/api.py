"""FastAPI application for OpenRSVP."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, time, timedelta
from hashlib import blake2s
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import tomllib

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException
from urllib.parse import urlencode

from .config import settings
from .crud import (
    create_event,
    create_rsvp,
    create_message,
    ensure_channel,
    get_channel_by_slug,
    get_public_channels,
    set_rsvp_status,
    touch_channel,
    update_event,
    update_rsvp,
    VALID_ATTENDANCE_STATUSES,
)
from .database import SessionLocal
from .models import Channel, Event, Message, Meta, RSVP
from .scheduler import start_scheduler, stop_scheduler
from .storage import fetch_root_token, init_db
from .utils import (
    duration_between,
    get_repo_url,
    humanize_time,
    render_markdown,
    utcnow,
)
from .web import register_web_routes
from .utils.ics import generate_ics

# Use uvicorn's error logger so messages get the level prefix in the default log
# format (needed for downstream filtering like Loki).
logger = logging.getLogger("uvicorn.error")


def _no_cache(response: Response) -> Response:
    """Prevent clients from caching dynamic pages so fresh data is shown."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


class EventFullError(Exception):
    """Raised when an event is at capacity for YES RSVPs."""


EVENT_FULL_ERROR = {
    "error": "EventFull",
    "message": "This event has reached its maximum number of attendees.",
}

RSVPS_CLOSED_ERROR = {
    "error": "RSVPsClosed",
    "message": "This organizer closed new RSVPs for this event.",
}


def _normalize_attendance_status(status: str | None) -> str:
    normalized = (status or "").strip().lower() or "yes"
    return normalized if normalized in VALID_ATTENDANCE_STATUSES else "maybe"


def _normalize_max_attendees(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="Invalid maximum attendees"
        ) from exc
    return value if value > 0 else None


def _load_app_version() -> str:
    """Return the current package version, falling back to pyproject for dev runs."""
    try:
        return pkg_version("openrsvp")
    except PackageNotFoundError:
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            data = tomllib.loads(pyproject_path.read_text())
            project = data.get("project") or {}
            return str(project.get("version") or "dev")
    return "dev"


APP_VERSION = _load_app_version()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()


app = FastAPI(title="OpenRSVP", version=APP_VERSION, lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Register web routes
register_web_routes(app)


def _asset_version(filename: str) -> str:
    file_path = static_dir / filename
    if not file_path.is_file():
        return APP_VERSION
    hasher = blake2s()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    digest = hasher.hexdigest()[:12]
    mtime = int(file_path.stat().st_mtime)
    return f"{digest}{mtime}"


ASSET_FILES = [
    "app.css",
    "app.js",
    "theme-init.js",
    "event-page.js",
    "event-admin.js",
    "qrcode.min.js",
]
ASSET_VERSIONS = {name: _asset_version(name) for name in ASSET_FILES}


def asset_version(filename: str) -> str:
    return ASSET_VERSIONS.get(filename, APP_VERSION)


templates.env.globals["app_version"] = APP_VERSION
templates.env.globals["repo_url"] = get_repo_url()
templates.env.globals["asset_version"] = asset_version
templates.env.filters["relative_time"] = humanize_time
templates.env.filters["duration"] = duration_between
templates.env.filters["markdown"] = render_markdown


def can_view_location(event: Event, rsvp: object = None) -> bool:
    if not getattr(event, "admin_approval_required", False):
        return True
    approval_status = getattr(rsvp, "approval_status", None)
    attendance_status = getattr(rsvp, "attendance_status", None)
    return approval_status == "approved" and attendance_status == "yes"


def event_location_text(event: Event, rsvp: object = None) -> str:
    if can_view_location(event, rsvp):
        return event.location or "Location TBD"
    return "Location hidden until your RSVP is approved"


def list_location_text(event: Event) -> str:
    if getattr(event, "admin_approval_required", False):
        return "Location hidden"
    return event.location or "Location TBD"


templates.env.globals["can_view_location"] = can_view_location
templates.env.globals["event_location_text"] = event_location_text
templates.env.globals["list_location_text"] = list_location_text

ADMIN_EVENTS_PER_PAGE = settings.admin_events_per_page
EVENTS_PER_PAGE = settings.events_per_page
RECENT_EVENTS_LIMIT = 5
CHANNEL_SUGGESTION_LIMIT = 12
HOME_CHANNEL_SELECT_LIMIT = 100
DISCOVER_CHANNELS_PER_PAGE = 50


def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _parse_datetime(raw: str) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid datetime") from exc


def _parse_iso_datetime_param(name: str, raw: str | None) -> datetime | None:
    """Parse an ISO8601 datetime query parameter or raise a 400."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid {name}; use ISO8601 format"
        ) from exc


def _local_to_utc(dt: datetime, offset_minutes: int) -> datetime:
    """Normalize a naive local datetime to UTC (still naive)."""
    try:
        offset = int(offset_minutes)
    except (TypeError, ValueError):
        offset = 0
    return dt + timedelta(minutes=offset)


def _wants_json(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    return request.url.path.startswith("/api/") or (
        "application/json" in accept and "text/html" not in accept
    )


def _render_error(request: Request, status_code: int, message: str | None):
    lower_message = (message or "").lower()
    extra_hint = None
    if "rsvp not found" in lower_message:
        extra_hint = "The event owner may have deleted this RSVP."
    context = {
        "request": request,
        "status_code": status_code,
        "message": None,
        "error_message": message or "Something went wrong.",
        "extra_hint": extra_hint,
    }
    return templates.TemplateResponse(
        request, "error.html", context, status_code=status_code
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Render HTTP errors as friendly pages unless JSON was requested."""
    if _wants_json(request):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    detail = exc.detail if isinstance(exc.detail, str) else "Something went wrong."
    return _render_error(request, exc.status_code, detail)


@app.exception_handler(OperationalError)
async def operational_error_handler(request: Request, exc: OperationalError):
    raw = str(exc.orig) if getattr(exc, "orig", None) else str(exc)
    lower = raw.lower()
    if "database is locked" in lower:
        logger.error(
            "SQLite database is locked while handling %s %s",
            request.method,
            request.url.path,
        )
        detail = "The database is busy at the moment. Please wait a few seconds and try again."
        status = 503
    else:
        logger.error(
            "Operational database error on %s %s: %s",
            request.method,
            request.url.path,
            raw,
        )
        detail = "We hit a database issue. Please try again."
        status = 500
    if _wants_json(request):
        return JSONResponse({"detail": detail}, status_code=status)
    return _render_error(request, status, detail)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    if _wants_json(request):
        return JSONResponse({"detail": exc.errors()}, status_code=422)
    return _render_error(
        request,
        422,
        "Some of the fields were invalid. Please double-check and try again.",
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Gracefully handle unexpected errors."""
    logger.exception(
        "Unhandled error while processing %s %s", request.method, request.url.path
    )
    if _wants_json(request):
        return JSONResponse({"detail": "Internal server error"}, status_code=500)
    return _render_error(
        request,
        500,
        "We hit a snag while processing that request. Please try again.",
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(static_dir / "favicon.ico")


@app.get("/favicon.svg", include_in_schema=False)
async def favicon_svg():
    return FileResponse(static_dir / "favicon.svg")


@app.get("/favicon-96x96.png", include_in_schema=False)
@app.get("/favicon.png", include_in_schema=False)
async def favicon_png():
    return FileResponse(static_dir / "favicon-96x96.png")


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def apple_touch_icon():
    return FileResponse(static_dir / "apple-touch-icon.png")


@app.get("/android-chrome-192x192.png", include_in_schema=False)
@app.get("/web-app-manifest-192x192.png", include_in_schema=False)
async def web_app_icon_192():
    return FileResponse(static_dir / "web-app-manifest-192x192.png")


@app.get("/android-chrome-512x512.png", include_in_schema=False)
@app.get("/web-app-manifest-512x512.png", include_in_schema=False)
async def web_app_icon_512():
    return FileResponse(static_dir / "web-app-manifest-512x512.png")


@app.get("/site.webmanifest", include_in_schema=False)
async def site_webmanifest(request: Request):
    """Expose a lightweight manifest for Android and modern browsers."""
    manifest = {
        "name": "OpenRSVP",
        "short_name": "OpenRSVP",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#f4f4f7",
        "theme_color": "#1e66f5",
        "icons": [
            {
                "src": str(request.url_for("web_app_icon_192")),
                "sizes": "192x192",
                "type": "image/png",
            },
            {
                "src": str(request.url_for("web_app_icon_512")),
                "sizes": "512x512",
                "type": "image/png",
            },
        ],
    }
    headers = {"Cache-Control": "public, max-age=3600"}
    return JSONResponse(manifest, headers=headers)


def _apply_pending_rsvp_close(event: Event, db: Session) -> None:
    if event.rsvps_closed or not event.rsvp_close_at or event.rsvp_close_at > utcnow():
        return
    scheduled_close = event.rsvp_close_at
    event.rsvps_closed = True
    event.rsvp_close_at = None
    db.add(event)
    logger.info(
        "Auto-close job toggled RSVPs closed for event %s (%s) scheduled for %s",
        event.id,
        event.title,
        scheduled_close.isoformat() if scheduled_close else "unknown",
    )


def _sync_rsvp_close_states(events: Iterable[Event], db: Session) -> None:
    for event in events:
        _apply_pending_rsvp_close(event, db)


def _ensure_event(db: Session, event_id: str) -> Event:
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    _apply_pending_rsvp_close(event, db)
    return event


def _ensure_rsvp(db: Session, event: Event, token: str) -> RSVP:
    stmt = select(RSVP).where(RSVP.event_id == event.id, RSVP.rsvp_token == token)
    rsvp = db.scalars(stmt).first()
    if not rsvp:
        raise HTTPException(status_code=404, detail="RSVP not found")
    return rsvp


def _ensure_rsvp_by_id(db: Session, event: Event, rsvp_id: str) -> RSVP:
    rsvp = db.get(RSVP, rsvp_id)
    if not rsvp or rsvp.event_id != event.id:
        raise HTTPException(status_code=404, detail="RSVP not found")
    return rsvp


def _yes_party_size(event: Event, *, exclude_rsvp: RSVP | None = None) -> int:
    excluded_id = exclude_rsvp.id if exclude_rsvp else None
    return sum(
        (rsvp.guest_count or 0) + 1
        for rsvp in event.rsvps
        if rsvp.attendance_status == "yes"
        and (excluded_id is None or rsvp.id != excluded_id)
    )


def _available_yes_slots(event: Event, current_rsvp: RSVP | None = None) -> int | None:
    if event.max_attendees is None:
        return None
    yes_party_size = _yes_party_size(event, exclude_rsvp=current_rsvp)
    return max(event.max_attendees - yes_party_size, 0)


def _enforce_attendance_capacity(
    *,
    event: Event,
    new_status: str,
    current_rsvp: RSVP | None = None,
    admin_token: str | None = None,
    force_accept: bool = False,
    new_guest_count: int | None = 0,
) -> bool:
    normalized_status = _normalize_attendance_status(new_status)
    if normalized_status != "yes":
        return False
    if event.max_attendees is None:
        return False
    yes_party_size = _yes_party_size(event, exclude_rsvp=current_rsvp)
    proposed_party_size = (_clamp_guest_count(new_guest_count) or 0) + 1
    if yes_party_size + proposed_party_size <= event.max_attendees:
        return False
    if force_accept:
        if not admin_token:
            raise HTTPException(status_code=403, detail="Invalid admin token")
        _require_admin_or_root(event, admin_token)
        return True
    raise EventFullError


def _event_full_response(request: Request, template: str | None, context: dict):
    if _wants_json(request):
        return JSONResponse(EVENT_FULL_ERROR, status_code=400)
    if template:
        payload = dict(context)
        payload["message"] = EVENT_FULL_ERROR["message"]
        payload["message_class"] = "alert-danger"
        return templates.TemplateResponse(
            request,
            template,
            payload,
            status_code=400,
        )
    raise HTTPException(status_code=400, detail=EVENT_FULL_ERROR["message"])


def _event_closed_response(
    request: Request, template: str | None, context: dict | None = None
):
    if _wants_json(request):
        return JSONResponse(RSVPS_CLOSED_ERROR, status_code=403)
    if template:
        payload = dict(context or {})
        payload["message"] = RSVPS_CLOSED_ERROR["message"]
        payload["message_class"] = "alert-danger"
        return templates.TemplateResponse(
            request,
            template,
            payload,
            status_code=403,
        )
    raise HTTPException(status_code=403, detail=RSVPS_CLOSED_ERROR["message"])


def _require_admin_or_root(event: Event, admin_token: str) -> None:
    if admin_token == event.admin_token:
        return
    stored_root = fetch_root_token()
    if admin_token != stored_root:
        raise HTTPException(status_code=403, detail="Invalid admin token")


def _require_root_access(root_token: str) -> None:
    stored = fetch_root_token()
    if root_token != stored:
        raise HTTPException(status_code=403, detail="Forbidden")


def _get_bearer_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header.split(" ", 1)[1].strip()
    return token or None


def _clamp_guest_count(raw: int | None) -> int:
    try:
        value = int(raw or 0)
    except (TypeError, ValueError):
        value = 0
    return max(0, min(value, 5))


def _normalize_event_times(
    *,
    start_time: str | None,
    end_time: str | None,
    timezone_offset_minutes: int,
):
    normalized_start = None
    normalized_end = None
    if start_time:
        parsed_start = _parse_datetime(start_time)
        normalized_start = _local_to_utc(parsed_start, timezone_offset_minutes)
    if end_time:
        parsed_end = _parse_datetime(end_time)
        normalized_end = _local_to_utc(parsed_end, timezone_offset_minutes)
    if normalized_start and normalized_end and normalized_end <= normalized_start:
        raise HTTPException(
            status_code=400, detail="End time must be after the start time"
        )
    return normalized_start, normalized_end


def _normalize_optional_close_time(
    close_time: str | None, timezone_offset_minutes: int
) -> datetime | None:
    if close_time is None:
        return None
    if isinstance(close_time, str) and not close_time.strip():
        return None
    parsed_close = _parse_datetime(close_time)
    return _local_to_utc(parsed_close, timezone_offset_minutes)


def _serialize_channel(channel: Channel):
    return {
        "id": channel.id,
        "name": channel.name,
        "slug": channel.slug,
        "visibility": channel.visibility,
        "score": channel.score,
        "created_at": channel.created_at.isoformat(),
        "last_used_at": channel.last_used_at.isoformat(),
    }


def _serialize_message(message: Message):
    return {
        "id": message.id,
        "event_id": message.event_id,
        "rsvp_id": message.rsvp_id,
        "author_id": message.author_id,
        "message_type": message.message_type,
        "visibility": message.visibility,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }


def _serialize_rsvp(
    rsvp: RSVP,
    *,
    include_token: bool = False,
    include_internal_id: bool = False,
    include_messages: list[Message] | None = None,
):
    payload = {
        "event_id": rsvp.event_id,
        "name": rsvp.name,
        "pronouns": rsvp.pronouns,
        "status": rsvp.approval_status,
        "approval_status": rsvp.approval_status,
        "attendance_status": rsvp.attendance_status,
        "guest_count": rsvp.guest_count,
        "is_private": rsvp.is_private,
        "created_at": rsvp.created_at.isoformat(),
        "last_modified": rsvp.last_modified.isoformat(),
    }
    if include_token:
        payload["rsvp_token"] = rsvp.rsvp_token
    if include_internal_id:
        payload["id"] = rsvp.id
    if include_messages is not None:
        payload["messages"] = [_serialize_message(m) for m in include_messages]
    return payload


def _serialize_event(
    event: Event,
    *,
    include_rsvps: list[RSVP] | None = None,
    include_private_counts: bool = False,
    include_admin_link: bool = False,
    include_location: bool = True,
    include_private_channel: bool = False,
    include_unapproved_yes_count: bool = True,
    include_messages: list[Message] | None = None,
):
    yes_count = event.yes_count
    if event.admin_approval_required and not include_unapproved_yes_count:
        yes_count = sum(
            (r.guest_count or 0) + 1
            for r in event.rsvps
            if r.approval_status == "approved" and r.attendance_status == "yes"
        )
    payload = {
        "id": event.id,
        "title": event.title,
        "description": event.description,
        "start_time": event.start_time.isoformat(),
        "end_time": event.end_time.isoformat() if event.end_time else None,
        "location": event.location if include_location else None,
        "is_private": event.is_private,
        "admin_approval_required": event.admin_approval_required,
        "score": event.score,
        "max_attendees": event.max_attendees,
        "yes_count": yes_count,
        "rsvps_closed": event.rsvps_closed,
        "rsvp_close_at": event.rsvp_close_at.isoformat()
        if event.rsvp_close_at
        else None,
        "created_at": event.created_at.isoformat(),
        "last_modified": event.last_modified.isoformat(),
        "last_accessed": event.last_accessed.isoformat(),
        "links": {
            "public": f"/e/{event.id}",
        },
    }
    if event.channel:
        if event.channel.visibility != "private" or include_private_channel:
            payload["channel"] = {
                "id": event.channel.id,
                "name": event.channel.name,
                "slug": event.channel.slug,
                "visibility": event.channel.visibility,
            }
    if include_admin_link:
        payload["links"]["admin"] = f"/e/{event.id}/admin/{event.admin_token}"
    if include_rsvps is not None:
        payload["rsvps"] = [_serialize_rsvp(r) for r in include_rsvps]
    if include_private_counts:
        approved = [r for r in event.rsvps if r.approval_status == "approved"]
        private_rsvps = [r for r in approved if r.is_private]
        public_rsvps = [r for r in approved if not r.is_private]
        payload["rsvp_counts"] = {
            "public": len(public_rsvps),
            "private": len(private_rsvps),
            "public_party_size": sum((r.guest_count or 0) + 1 for r in public_rsvps),
            "private_party_size": sum((r.guest_count or 0) + 1 for r in private_rsvps),
        }
    if include_messages is not None:
        payload["messages"] = [_serialize_message(m) for m in include_messages]
    return payload


def _rsvp_stats(rsvps: list[RSVP]) -> dict[str, int]:
    approved = [r for r in rsvps if r.approval_status == "approved"]
    yes_rsvps = [r for r in approved if r.attendance_status == "yes"]
    maybe_rsvps = [r for r in approved if r.attendance_status == "maybe"]
    yes_guest_count = sum(r.guest_count for r in yes_rsvps)
    maybe_guest_count = sum(r.guest_count for r in maybe_rsvps)
    no_rsvps = [r for r in approved if r.attendance_status == "no"]
    no_guest_count = sum(r.guest_count for r in no_rsvps)
    return {
        "yes_count": len(yes_rsvps),
        "yes_guest_count": yes_guest_count,
        "yes_total": len(yes_rsvps) + yes_guest_count,
        "maybe_count": len(maybe_rsvps),
        "maybe_guest_count": maybe_guest_count,
        "maybe_total": len(maybe_rsvps) + maybe_guest_count,
        "no_count": len(no_rsvps),
        "no_guest_count": no_guest_count,
        "no_total": len(no_rsvps) + no_guest_count,
    }


def _sorted_messages(messages: Iterable[Message]) -> list[Message]:
    return sorted(messages, key=lambda msg: msg.created_at, reverse=True)


def _event_messages(event: Event, *, visibilities: set[str]) -> list[Message]:
    return _sorted_messages([m for m in event.messages if m.visibility in visibilities])


def _rsvp_messages(rsvp: RSVP, *, visibilities: set[str]) -> list[Message]:
    return _sorted_messages([m for m in rsvp.messages if m.visibility in visibilities])


def _create_message_if_content(
    db: Session,
    *,
    event: Event | None,
    rsvp: RSVP | None,
    author_id: str | None,
    message_type: str,
    visibility: str,
    content: str | None,
):
    content = (content or "").strip()
    if not content:
        return None
    return create_message(
        db,
        event=event,
        rsvp=rsvp,
        author_id=author_id,
        message_type=message_type,
        visibility=visibility,
        content=content,
    )


def _log_rsvp_created(db: Session, *, event: Event, rsvp: RSVP):
    guest_count = rsvp.guest_count or 0
    guest_phrase = (
        "no guests"
        if guest_count == 0
        else f"{guest_count} guest{'s' if guest_count != 1 else ''}"
    )
    attendance = rsvp.attendance_status.capitalize()
    approval = rsvp.approval_status
    content = f"RSVP created ({attendance}; approval {approval}; {guest_phrase})"
    return create_message(
        db,
        event=event,
        rsvp=rsvp,
        author_id=None,
        message_type="rsvp_created",
        visibility="admin",
        content=content,
    )


def _log_force_accept(db: Session, *, event: Event, rsvp: RSVP):
    spots = (
        f"{event.yes_count}/{event.max_attendees}"
        if event.max_attendees is not None
        else f"{event.yes_count}"
    )
    return create_message(
        db,
        event=event,
        rsvp=rsvp,
        author_id=None,
        message_type="force_accept",
        visibility="admin",
        content=f"Force accepted RSVP for {rsvp.name} (capacity {spots})",
    )


def _apply_rsvp_status_change(
    db: Session,
    *,
    event: Event,
    rsvp: RSVP,
    new_status: str,
    reason: str | None = None,
) -> RSVP:
    """Update approval status and log audit + attendee messaging."""
    set_rsvp_status(db, rsvp=rsvp, status=new_status)
    create_message(
        db,
        event=event,
        rsvp=rsvp,
        author_id=None,
        message_type="rsvp_status_change",
        visibility="admin",
        content=f"RSVP marked as {rsvp.approval_status}",
    )
    if new_status == "rejected":
        rejection_content = reason or "This RSVP was rejected by the organizer."
        create_message(
            db,
            event=event,
            rsvp=rsvp,
            author_id=None,
            message_type="rejection_reason",
            visibility="attendee",
            content=rejection_content,
        )
    return rsvp


class EventCreatePayload(BaseModel):
    title: str
    description: str | None = None
    start_time: str = Field(..., description="ISO datetime string")
    end_time: str | None = Field(
        None, description="Optional ISO datetime string after start_time"
    )
    timezone_offset_minutes: int = 0
    location: str | None = None
    channel_name: str | None = None
    channel_visibility: str = "public"
    is_private: bool = False
    admin_approval_required: bool = False
    max_attendees: int | None = Field(
        None, ge=1, description="Maximum number of YES RSVPs allowed"
    )
    rsvps_closed: bool = False
    rsvp_close_at: str | None = Field(
        None, description="Optional ISO datetime string for auto-closing RSVPs"
    )


class EventUpdatePayload(BaseModel):
    title: str | None = None
    description: str | None = None
    start_time: str | None = Field(None, description="ISO datetime string")
    end_time: str | None = Field(None, description="Optional ISO datetime string")
    timezone_offset_minutes: int | None = None
    location: str | None = None
    channel_name: str | None = None
    channel_visibility: str | None = None
    is_private: bool | None = None
    admin_approval_required: bool | None = None
    max_attendees: int | None = Field(
        None, ge=1, description="Maximum number of YES RSVPs allowed"
    )
    rsvps_closed: bool | None = None
    rsvp_close_at: str | None = Field(
        None, description="Optional ISO datetime string for auto-closing RSVPs"
    )


class RSVPCreatePayload(BaseModel):
    name: str
    attendance_status: str
    pronouns: str | None = None
    guest_count: int = 0
    note: str | None = None
    is_private_rsvp: bool = False


class RSVPUpdatePayload(BaseModel):
    name: str | None = None
    attendance_status: str | None = None
    pronouns: str | None = None
    guest_count: int | None = None
    note: str | None = None
    is_private_rsvp: bool | None = None


class MessageCreatePayload(BaseModel):
    content: str
    visibility: str
    message_type: str


class AttendeeMessagePayload(BaseModel):
    content: str


class StatusChangePayload(BaseModel):
    reason: str | None = None


def _channel_error_message(raw: str) -> str:
    message = raw.lower()
    if "different visibility" in message:
        return (
            "That channel name already exists but with a different visibility. "
            "Pick another channel name or match the visibility of the existing channel."
        )
    if "invalid channel name" in message:
        return "Channel names must include at least one letter or number."
    if "invalid channel visibility" in message:
        return "Channel visibility must be set to Public or Private."
    return "We couldn't save that channel. Please try again with a different name."


def _require_admin_header(event: Event, request: Request) -> str:
    token = _get_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    _require_admin_or_root(event, token)
    return token


def _require_rsvp_from_header(event: Event, request: Request, db: Session) -> RSVP:
    token = _get_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return _ensure_rsvp(db, event, token)


def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request", "").lower() == "true"


def _token_is_admin_or_root(
    event: Event, *, token: str | None, root_token: str | None
) -> bool:
    if not token:
        return False
    if token == event.admin_token:
        return True
    return bool(root_token and token == root_token)


def _can_view_event_location_api(
    db: Session,
    *,
    event: Event,
    token: str | None,
    root_token: str | None,
) -> bool:
    if not event.admin_approval_required:
        return True
    if _token_is_admin_or_root(event, token=token, root_token=root_token):
        return True
    if not token:
        return False
    stmt = select(RSVP).where(RSVP.event_id == event.id, RSVP.rsvp_token == token)
    rsvp = db.scalar(stmt)
    return bool(
        rsvp and rsvp.approval_status == "approved" and rsvp.attendance_status == "yes"
    )


def _fetch_rsvp_by_token_in_session(db: Session, token: str | None) -> RSVP | None:
    if not token:
        return None
    stmt = select(RSVP).where(RSVP.rsvp_token == token)
    return db.scalar(stmt)


def _can_view_event_location_list(
    event: Event,
    *,
    token: str | None,
    root_token: str | None,
    rsvp: RSVP | None,
) -> bool:
    if not event.admin_approval_required:
        return True
    if _token_is_admin_or_root(event, token=token, root_token=root_token):
        return True
    if not token:
        return False
    if rsvp and rsvp.event_id == event.id:
        return rsvp.approval_status == "approved" and rsvp.attendance_status == "yes"
    return False


def _can_view_private_channel(
    event: Event, *, token: str | None, root_token: str | None
) -> bool:
    return _token_is_admin_or_root(event, token=token, root_token=root_token)


def _fetch_root_token_in_session(db: Session) -> str | None:
    meta = db.get(Meta, settings.root_token_key)
    return meta.value if meta else None


def _approval_counts(rsvps: Sequence[RSVP]) -> dict[str, int]:
    return {
        "approved": sum(1 for r in rsvps if r.approval_status == "approved"),
        "pending": sum(1 for r in rsvps if r.approval_status == "pending"),
        "rejected": sum(1 for r in rsvps if r.approval_status == "rejected"),
    }


def _admin_rsvp_partial_response(
    request: Request,
    *,
    event: Event,
    admin_token: str,
    rsvp: RSVP,
):
    rsvps = list(event.rsvps)
    approval_counts = _approval_counts(rsvps)
    available_yes_slots = _available_yes_slots(event)
    event_is_full = (
        available_yes_slots == 0 if available_yes_slots is not None else False
    )
    messages = _rsvp_messages(rsvp, visibilities={"public", "attendee", "admin"})
    response = templates.TemplateResponse(
        request,
        "partials/admin_rsvp_response.html",
        {
            "request": request,
            "event": event,
            "admin_token": admin_token,
            "rsvp": rsvp,
            "messages": messages,
            "approval_counts": approval_counts,
            "rsvp_total": len(rsvps),
            "event_is_full": event_is_full,
        },
    )
    return _no_cache(response)


def _attendee_messages_partial_response(
    request: Request,
    *,
    event: Event,
    rsvp: RSVP,
    attendee_message_flash: str | None = None,
    attendee_message_class: str | None = None,
):
    attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
    response = templates.TemplateResponse(
        request,
        "partials/attendee_messages.html",
        {
            "request": request,
            "event": event,
            "rsvp": rsvp,
            "attendee_messages": attendee_messages,
            "attendee_message_flash": attendee_message_flash,
            "attendee_message_class": attendee_message_class,
        },
    )
    return _no_cache(response)


def _channel_search_clause(query: str | None):
    if not query:
        return None
    cleaned = query.strip()
    if not cleaned:
        return None
    like = f"%{cleaned}%"
    return or_(Channel.name.ilike(like), Channel.slug.ilike(like))


def _event_search_clause(query: str | None):
    if not query:
        return None
    cleaned = query.strip()
    if not cleaned:
        return None
    like = f"%{cleaned}%"
    return or_(
        Event.title.ilike(like),
        Event.description.ilike(like),
        Event.location.ilike(like),
        Event.id.ilike(like),
    )


def _fetch_channels(db: Session, query: str | None):
    clause = _channel_search_clause(query)
    stmt = select(Channel).order_by(Channel.name.asc())
    if clause is not None:
        stmt = stmt.where(clause)
    channels = db.scalars(stmt).all()
    for channel in channels:
        _ = list(channel.events)
    return channels


def _build_pagination(
    *,
    page: int,
    per_page: int,
    total_events: int,
    include_query: bool,
    query: str | None,
):
    total_pages = (
        max(1, (total_events + per_page - 1) // per_page) if total_events else 1
    )
    page = max(1, min(page, total_pages)) if total_events else 1
    pagination = {
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_events": total_events,
        "has_prev": page > 1,
        "has_next": page < total_pages and total_events > 0,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < total_pages and total_events > 0 else None,
    }
    if include_query:
        pagination["query"] = query or ""
    return pagination


def paginate_events(
    db: Session,
    *,
    filters: Iterable | None,
    order_by,
    per_page: int,
    page: int,
    include_query: bool = False,
    query: str | None = None,
    include_rsvps: bool = False,
):
    filters = list(filters or [])
    count_stmt = select(func.count()).select_from(Event)
    for condition in filters:
        count_stmt = count_stmt.where(condition)
    total_events = db.scalar(count_stmt) or 0
    pagination = _build_pagination(
        page=page,
        per_page=per_page,
        total_events=total_events,
        include_query=include_query,
        query=query,
    )
    page_number = pagination["page"]
    offset = (page_number - 1) * per_page if total_events else 0

    stmt = select(Event)
    if isinstance(order_by, (list, tuple)):
        stmt = stmt.order_by(*order_by)
    else:
        stmt = stmt.order_by(order_by)
    for condition in filters:
        stmt = stmt.where(condition)
    stmt = stmt.offset(offset).limit(per_page)
    events = db.scalars(stmt).all()
    _sync_rsvp_close_states(events, db)
    if include_rsvps:
        for event in events:
            _ = list(event.rsvps)
    return events, pagination


def paginate_channels(
    db: Session,
    *,
    query: str | None,
    page: int,
    per_page: int = ADMIN_EVENTS_PER_PAGE,
    include_events: bool = True,
):
    clause = _channel_search_clause(query)
    count_stmt = select(func.count()).select_from(Channel)
    if clause is not None:
        count_stmt = count_stmt.where(clause)
    total_channels = db.scalar(count_stmt) or 0
    pagination = _build_pagination(
        page=page,
        per_page=per_page,
        total_events=total_channels,
        include_query=True,
        query=query,
    )
    page_number = pagination["page"]
    offset = (page_number - 1) * per_page if total_channels else 0
    stmt = select(Channel).order_by(Channel.name.asc())
    if clause is not None:
        stmt = stmt.where(clause)
    stmt = stmt.offset(offset).limit(per_page)
    channels = db.scalars(stmt).all()
    if include_events:
        for channel in channels:
            _ = list(channel.events)
    return channels, pagination


def _paginate_public_channels(
    db: Session,
    *,
    query: str | None,
    page: int,
    per_page: int = ADMIN_EVENTS_PER_PAGE,
):
    clause = _channel_search_clause(query)
    count_stmt = (
        select(func.count()).select_from(Channel).where(Channel.visibility == "public")
    )
    if clause is not None:
        count_stmt = count_stmt.where(clause)
    total_channels = db.scalar(count_stmt) or 0
    pagination = _build_pagination(
        page=page,
        per_page=per_page,
        total_events=total_channels,
        include_query=True,
        query=query,
    )
    page_number = pagination["page"]
    offset = (page_number - 1) * per_page if total_channels else 0
    stmt = (
        select(Channel)
        .where(Channel.visibility == "public")
        .order_by(Channel.name.asc())
        .offset(offset)
        .limit(per_page)
    )
    if clause is not None:
        stmt = stmt.where(clause)
    channels = db.scalars(stmt).all()
    return channels, pagination


def _visible_event_filters(
    *,
    include_private: bool,
    channel_id: str | None,
    now: datetime,
):
    filters = [
        or_(
            Event.score >= settings.hide_threshold,
            Event.created_at >= now - timedelta(days=settings.hide_after_days),
        )
    ]
    if not include_private:
        filters.append(Event.is_private.is_(False))
    if channel_id:
        filters.append(Event.channel_id == channel_id)
    return filters


def _paginate_visible_events(
    db: Session,
    *,
    include_private: bool,
    channel_id: str | None,
    page: int,
    per_page: int = EVENTS_PER_PAGE,
):
    now = utcnow()
    filters = _visible_event_filters(
        include_private=include_private,
        channel_id=channel_id,
        now=now,
    )
    return paginate_events(
        db,
        filters=filters,
        order_by=Event.start_time.desc(),
        per_page=per_page,
        page=page,
    )


def _paginate_upcoming_events(
    db: Session,
    *,
    include_private: bool,
    channel_id: str | None,
    page: int,
    per_page: int = EVENTS_PER_PAGE,
):
    now = utcnow()
    filters = _visible_event_filters(
        include_private=include_private,
        channel_id=channel_id,
        now=now,
    )
    filters.append(Event.start_time >= now)
    return paginate_events(
        db,
        filters=filters,
        order_by=Event.start_time.asc(),
        per_page=per_page,
        page=page,
    )


def _recent_events(
    db: Session,
    *,
    include_private: bool,
    channel_id: str | None,
    limit: int = RECENT_EVENTS_LIMIT,
):
    now = utcnow()
    filters = _visible_event_filters(
        include_private=include_private,
        channel_id=channel_id,
        now=now,
    )
    filters.append(Event.start_time < now)
    stmt = select(Event).order_by(Event.start_time.desc()).limit(limit)
    for condition in filters:
        stmt = stmt.where(condition)
    events = db.scalars(stmt).all()
    _sync_rsvp_close_states(events, db)
    return events


def _paginate_events(
    db: Session, *, page: int, query: str | None, include_rsvps: bool = False
):
    clause = _event_search_clause(query)
    filters = [clause] if clause is not None else []
    return paginate_events(
        db,
        filters=filters,
        order_by=Event.created_at.desc(),
        per_page=ADMIN_EVENTS_PER_PAGE,
        page=page,
        include_query=True,
        query=query,
        include_rsvps=include_rsvps,
    )


@app.get("/")
def homepage(
    request: Request,
    channel: str | None = Query(default=None),
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    if channel:
        slug_channel = get_channel_by_slug(db, channel)
        if not slug_channel:
            raise HTTPException(status_code=404, detail="Channel not found")
        if slug_channel.visibility == "private":
            return RedirectResponse(
                url=f"/channel/p/{slug_channel.slug}", status_code=303
            )
        return RedirectResponse(url=f"/channel/{slug_channel.slug}", status_code=303)
    upcoming_events, upcoming_pagination = _paginate_upcoming_events(
        db,
        include_private=False,
        channel_id=None,
        page=page,
        per_page=EVENTS_PER_PAGE,
    )
    recent_events = _recent_events(
        db,
        include_private=False,
        channel_id=None,
        limit=RECENT_EVENTS_LIMIT,
    )
    public_channels = get_public_channels(db, limit=HOME_CHANNEL_SELECT_LIMIT)
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "request": request,
            "upcoming_events": upcoming_events,
            "upcoming_pagination": upcoming_pagination,
            "recent_events": recent_events,
            "recent_limit": RECENT_EVENTS_LIMIT,
            "public_channels": public_channels,
        },
    )


@app.get("/event/create")
def event_create_page(request: Request, db: Session = Depends(get_db)):
    public_channels = get_public_channels(db, limit=CHANNEL_SUGGESTION_LIMIT)
    return templates.TemplateResponse(
        request,
        "event_create.html",
        {
            "request": request,
            "public_channels": public_channels,
        },
    )


@app.post("/events")
def submit_event(
    request: Request,
    title: str = Form(...),
    description: str | None = Form(None),
    start_time: str = Form(...),
    location: str | None = Form(None),
    channel_name: str | None = Form(None),
    channel_visibility: str = Form("public"),
    is_private: bool = Form(False),
    admin_approval_required: bool = Form(False),
    max_attendees: str | None = Form(None),
    end_time: str | None = Form(None),
    rsvps_closed: bool = Form(False),
    rsvp_close_at: str | None = Form(None),
    timezone_offset_minutes: int = Form(0),
    db: Session = Depends(get_db),
):
    public_channels = get_public_channels(db, limit=CHANNEL_SUGGESTION_LIMIT)
    channel = None
    cleaned_channel_name = channel_name.strip() if channel_name else ""
    if cleaned_channel_name:
        try:
            channel = ensure_channel(
                db, name=cleaned_channel_name, visibility=channel_visibility
            )
        except ValueError as exc:
            return templates.TemplateResponse(
                request,
                "event_create.html",
                {
                    "request": request,
                    "public_channels": public_channels,
                    "message": _channel_error_message(str(exc)),
                    "message_class": "alert-danger",
                },
                status_code=400,
            )
    parsed_start = _parse_datetime(start_time)
    normalized_start = _local_to_utc(parsed_start, timezone_offset_minutes)
    normalized_end = None
    if end_time:
        parsed_end = _parse_datetime(end_time)
        normalized_end = _local_to_utc(parsed_end, timezone_offset_minutes)
        if normalized_end <= normalized_start:
            return templates.TemplateResponse(
                request,
                "event_create.html",
                {
                    "request": request,
                    "public_channels": public_channels,
                    "message": "End time must be after the start time.",
                    "message_class": "alert-danger",
                },
                status_code=400,
            )
    normalized_close = _normalize_optional_close_time(
        rsvp_close_at, timezone_offset_minutes
    )
    normalized_close = _normalize_optional_close_time(
        rsvp_close_at, timezone_offset_minutes
    )
    try:
        normalized_max = _normalize_max_attendees(max_attendees)
    except HTTPException:
        return templates.TemplateResponse(
            request,
            "event_create.html",
            {
                "request": request,
                "public_channels": public_channels,
                "message": "Maximum attendees must be a positive number.",
                "message_class": "alert-danger",
            },
            status_code=400,
        )
    event = create_event(
        db,
        title=title,
        description=description,
        start_time=normalized_start,
        end_time=normalized_end,
        location=location,
        channel=channel,
        is_private=is_private,
        admin_approval_required=admin_approval_required,
        max_attendees=normalized_max,
        rsvps_closed=rsvps_closed,
        rsvp_close_at=normalized_close,
    )
    return templates.TemplateResponse(
        request,
        "event_created.html",
        {
            "request": request,
            "event": event,
            "admin_link": f"/e/{event.id}/admin/{event.admin_token}",
            "public_channels": public_channels,
        },
    )


@app.get("/e/{event_id}")
def event_page(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    event.last_accessed = utcnow()
    db.add(event)
    if event.channel:
        touch_channel(event.channel)
        db.add(event.channel)
    approved_rsvps = [r for r in event.rsvps if r.approval_status == "approved"]
    rsvps = [r for r in approved_rsvps if not getattr(r, "is_private", False)]
    private_rsvps = [r for r in approved_rsvps if getattr(r, "is_private", False)]
    private_rsvp_count = len(private_rsvps)
    public_party_size = sum((r.guest_count or 0) + 1 for r in rsvps)
    private_party_size = sum((r.guest_count or 0) + 1 for r in private_rsvps)
    available_yes_slots = _available_yes_slots(event)
    event_is_full = (
        available_yes_slots == 0 if available_yes_slots is not None else False
    )
    now = utcnow()
    event_end_at = event.end_time
    if event_end_at is None:
        start_reference = event.start_time or event.created_at
        next_day = start_reference.date() + timedelta(days=1)
        event_end_at = datetime.combine(next_day, time.min)
    event_is_over = now >= event_end_at
    message = request.query_params.get("message")
    message_class = request.query_params.get("message_class")
    public_messages = _event_messages(event, visibilities={"public"})
    response = templates.TemplateResponse(
        request,
        "event.html",
        {
            "request": request,
            "event": event,
            "rsvps": rsvps,
            "private_rsvp_count": private_rsvp_count,
            "public_party_size": public_party_size,
            "private_party_size": private_party_size,
            "available_yes_slots": available_yes_slots,
            "event_is_full": event_is_full,
            "event_is_over": event_is_over,
            "message": message,
            "message_class": message_class,
            "public_messages": public_messages,
        },
    )
    return _no_cache(response)


@app.get("/e/{event_id}/rsvp")
def rsvp_form(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    available_yes_slots = _available_yes_slots(event)
    event_is_full = (
        available_yes_slots == 0 if available_yes_slots is not None else False
    )
    return templates.TemplateResponse(
        request,
        "rsvp_form.html",
        {
            "request": request,
            "event": event,
            "available_yes_slots": available_yes_slots,
            "event_is_full": event_is_full,
        },
    )


@app.post("/e/{event_id}/rsvp")
def create_rsvp_view(
    event_id: str,
    request: Request,
    name: str = Form(...),
    attendance_status: str = Form(...),
    pronouns: str | None = Form(None),
    guest_count: int = Form(0),
    note: str | None = Form(None),
    is_private_rsvp: bool = Form(False),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    if event.rsvps_closed:
        available_yes_slots = _available_yes_slots(event)
        event_is_full = (
            available_yes_slots == 0 if available_yes_slots is not None else False
        )
        return _event_closed_response(
            request,
            "rsvp_form.html",
            {
                "request": request,
                "event": event,
                "available_yes_slots": available_yes_slots,
                "event_is_full": event_is_full,
            },
        )
    normalized_status = _normalize_attendance_status(attendance_status)
    guest_count = max(0, min(guest_count, 5))
    try:
        _enforce_attendance_capacity(
            event=event,
            new_status=normalized_status,
            new_guest_count=guest_count,
            force_accept=False,
        )
    except EventFullError:
        public_channels = get_public_channels(db, limit=CHANNEL_SUGGESTION_LIMIT)
        return _event_full_response(
            request,
            "rsvp_form.html",
            {
                "request": request,
                "event": event,
                "public_channels": public_channels,
                "available_yes_slots": _available_yes_slots(event),
                "event_is_full": True,
            },
        )
    rsvp = create_rsvp(
        db,
        event=event,
        name=name,
        attendance_status=normalized_status,
        pronouns=pronouns,
        guest_count=guest_count,
        is_private=is_private_rsvp,
    )
    _log_rsvp_created(db, event=event, rsvp=rsvp)
    _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=rsvp.id,
        message_type="user_note",
        visibility="attendee",
        content=note,
    )
    return templates.TemplateResponse(
        request,
        "rsvp_created.html",
        {
            "request": request,
            "event": event,
            "rsvp": rsvp,
            "magic_link": f"/e/{event.id}/rsvp/{rsvp.rsvp_token}",
        },
    )


@app.get("/e/{event_id}/rsvp/{rsvp_token}")
def edit_rsvp(
    event_id: str, rsvp_token: str, request: Request, db: Session = Depends(get_db)
):
    event = _ensure_event(db, event_id)
    rsvp = _ensure_rsvp(db, event, rsvp_token)
    available_yes_slots = _available_yes_slots(event, current_rsvp=rsvp)
    event_is_full = _available_yes_slots(event) == 0 if event.max_attendees else False
    attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
    return templates.TemplateResponse(
        request,
        "rsvp_edit.html",
        {
            "request": request,
            "event": event,
            "rsvp": rsvp,
            "attendee_messages": attendee_messages,
            "available_yes_slots": available_yes_slots,
            "event_is_full": event_is_full,
            "attendee_message_flash": None,
            "attendee_message_class": None,
        },
    )


@app.post("/e/{event_id}/rsvp/{rsvp_token}")
def save_rsvp(
    event_id: str,
    rsvp_token: str,
    request: Request,
    name: str = Form(...),
    attendance_status: str = Form(...),
    pronouns: str | None = Form(None),
    guest_count: int = Form(0),
    note: str | None = Form(None),
    is_private_rsvp: bool = Form(False),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    rsvp = _ensure_rsvp(db, event, rsvp_token)
    normalized_status = _normalize_attendance_status(attendance_status)
    guest_count = max(0, min(guest_count, 5))
    try:
        _enforce_attendance_capacity(
            event=event,
            new_status=normalized_status,
            current_rsvp=rsvp,
            new_guest_count=guest_count,
            force_accept=False,
        )
    except EventFullError:
        attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
        return _event_full_response(
            request,
            "rsvp_edit.html",
            {
                "request": request,
                "event": event,
                "rsvp": rsvp,
                "attendee_messages": attendee_messages,
                "available_yes_slots": _available_yes_slots(event, current_rsvp=rsvp),
                "event_is_full": True,
            },
        )
    update_rsvp(
        db,
        rsvp=rsvp,
        name=name,
        attendance_status=normalized_status,
        pronouns=pronouns,
        guest_count=guest_count,
        is_private=is_private_rsvp,
    )
    _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=rsvp.id,
        message_type="user_note",
        visibility="attendee",
        content=note,
    )
    attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
    available_yes_slots = _available_yes_slots(event, current_rsvp=rsvp)
    event_is_full = _available_yes_slots(event) == 0 if event.max_attendees else False
    return templates.TemplateResponse(
        request,
        "rsvp_edit.html",
        {
            "request": request,
            "event": event,
            "rsvp": rsvp,
            "attendee_messages": attendee_messages,
            "available_yes_slots": available_yes_slots,
            "event_is_full": event_is_full,
            "message": "RSVP updated",
            "message_class": "alert-success",
            "attendee_message_flash": None,
            "attendee_message_class": None,
        },
    )


@app.post("/e/{event_id}/rsvp/{rsvp_token}/messages")
def add_attendee_rsvp_message(
    event_id: str,
    rsvp_token: str,
    request: Request,
    content: str = Form(...),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    rsvp = _ensure_rsvp(db, event, rsvp_token)
    message = _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=rsvp.id,
        message_type="user_note",
        visibility="attendee",
        content=content,
    )
    attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
    available_yes_slots = _available_yes_slots(event, current_rsvp=rsvp)
    event_is_full = _available_yes_slots(event) == 0 if event.max_attendees else False
    attendee_message_flash = "Message sent" if message else "Message cannot be empty"
    attendee_message_class = "alert-success" if message else "alert-danger"
    if _is_htmx(request):
        return _attendee_messages_partial_response(
            request,
            event=event,
            rsvp=rsvp,
            attendee_message_flash=attendee_message_flash,
            attendee_message_class=attendee_message_class,
        )
    return templates.TemplateResponse(
        request,
        "rsvp_edit.html",
        {
            "request": request,
            "event": event,
            "rsvp": rsvp,
            "attendee_messages": attendee_messages,
            "available_yes_slots": available_yes_slots,
            "event_is_full": event_is_full,
            "attendee_message_flash": attendee_message_flash,
            "attendee_message_class": attendee_message_class,
        },
    )


@app.post("/e/{event_id}/rsvp/{rsvp_token}/delete")
def delete_rsvp_self(
    event_id: str, rsvp_token: str, request: Request, db: Session = Depends(get_db)
):
    event = _ensure_event(db, event_id)
    rsvp = _ensure_rsvp(db, event, rsvp_token)
    db.delete(rsvp)
    db.commit()
    params = urlencode(
        {"message": "Your RSVP has been deleted.", "message_class": "alert-success"}
    )
    return RedirectResponse(url=f"/e/{event.id}?{params}", status_code=303)


@app.get("/e/{event_id}/admin/{admin_token}")
def event_admin(
    event_id: str, admin_token: str, request: Request, db: Session = Depends(get_db)
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvps = list(event.rsvps)
    rsvp_stats = _rsvp_stats(rsvps)
    available_yes_slots = _available_yes_slots(event)
    event_is_full = (
        available_yes_slots == 0 if available_yes_slots is not None else False
    )
    approval_counts = _approval_counts(rsvps)
    public_channels = get_public_channels(db, limit=CHANNEL_SUGGESTION_LIMIT)
    message = request.query_params.get("message")
    message_class = request.query_params.get("message_class")
    admin_messages = _event_messages(
        event, visibilities={"public", "attendee", "admin"}
    )
    rsvp_messages = {
        rsvp.id: _rsvp_messages(rsvp, visibilities={"public", "attendee", "admin"})
        for rsvp in rsvps
    }
    response = templates.TemplateResponse(
        request,
        "event_admin.html",
        {
            "request": request,
            "event": event,
            "rsvps": rsvps,
            "rsvp_stats": rsvp_stats,
            "available_yes_slots": available_yes_slots,
            "event_is_full": event_is_full,
            "approval_counts": approval_counts,
            "public_channels": public_channels,
            "admin_token": admin_token,
            "message": message,
            "message_class": message_class,
            "admin_messages": admin_messages,
            "rsvp_messages": rsvp_messages,
        },
    )
    return _no_cache(response)


@app.post("/e/{event_id}/admin/{admin_token}/rsvp/{rsvp_id}/delete")
def delete_rsvp_admin(
    event_id: str,
    admin_token: str,
    rsvp_id: str,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvp = db.get(RSVP, rsvp_id)
    if not rsvp or rsvp.event_id != event.id:
        raise HTTPException(status_code=404, detail="RSVP not found")
    db.delete(rsvp)
    db.commit()
    params = urlencode({"message": "RSVP removed.", "message_class": "alert-success"})
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
    )


@app.post("/e/{event_id}/admin/{admin_token}")
def save_event_admin(
    event_id: str,
    admin_token: str,
    request: Request,
    title: str = Form(...),
    description: str | None = Form(None),
    start_time: str = Form(...),
    location: str | None = Form(None),
    channel_name: str | None = Form(None),
    channel_visibility: str = Form("public"),
    is_private: bool = Form(False),
    admin_approval_required: bool = Form(False),
    rsvps_closed: bool = Form(False),
    max_attendees: str | None = Form(None),
    end_time: str | None = Form(None),
    rsvp_close_at: str | None = Form(None),
    timezone_offset_minutes: int = Form(0),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvps = list(event.rsvps)
    rsvp_stats = _rsvp_stats(rsvps)
    available_yes_slots = _available_yes_slots(event)
    event_is_full = (
        available_yes_slots == 0 if available_yes_slots is not None else False
    )
    approval_counts = {
        "approved": sum(1 for r in rsvps if r.approval_status == "approved"),
        "pending": sum(1 for r in rsvps if r.approval_status == "pending"),
        "rejected": sum(1 for r in rsvps if r.approval_status == "rejected"),
    }
    admin_messages = _event_messages(
        event, visibilities={"public", "attendee", "admin"}
    )
    rsvp_messages = {
        rsvp.id: _rsvp_messages(rsvp, visibilities={"public", "attendee", "admin"})
        for rsvp in rsvps
    }
    parsed_start = _parse_datetime(start_time)
    normalized_start = _local_to_utc(parsed_start, timezone_offset_minutes)
    normalized_end = None
    if end_time:
        parsed_end = _parse_datetime(end_time)
        normalized_end = _local_to_utc(parsed_end, timezone_offset_minutes)
        if normalized_end <= normalized_start:
            rsvps = list(event.rsvps)
            public_channels = get_public_channels(db, limit=CHANNEL_SUGGESTION_LIMIT)
            return templates.TemplateResponse(
                request,
                "event_admin.html",
                {
                    "request": request,
                    "event": event,
                    "rsvps": rsvps,
                    "rsvp_stats": rsvp_stats,
                    "available_yes_slots": available_yes_slots,
                    "event_is_full": event_is_full,
                    "approval_counts": approval_counts,
                    "admin_messages": admin_messages,
                    "rsvp_messages": rsvp_messages,
                    "public_channels": public_channels,
                    "admin_token": admin_token,
                    "message": "End time must be after the start time.",
                    "message_class": "alert-danger",
                },
                status_code=400,
            )
    normalized_close = _normalize_optional_close_time(
        rsvp_close_at, timezone_offset_minutes
    )
    try:
        normalized_max = _normalize_max_attendees(max_attendees)
    except HTTPException:
        rsvps = list(event.rsvps)
        public_channels = get_public_channels(db, limit=CHANNEL_SUGGESTION_LIMIT)
        return templates.TemplateResponse(
            request,
            "event_admin.html",
            {
                "request": request,
                "event": event,
                "rsvps": rsvps,
                "rsvp_stats": rsvp_stats,
                "available_yes_slots": available_yes_slots,
                "event_is_full": event_is_full,
                "approval_counts": approval_counts,
                "admin_messages": admin_messages,
                "rsvp_messages": rsvp_messages,
                "public_channels": public_channels,
                "admin_token": admin_token,
                "message": "Maximum attendees must be a positive number.",
                "message_class": "alert-danger",
            },
            status_code=400,
        )
    cleaned_channel_name = channel_name.strip() if channel_name else ""
    public_channels = get_public_channels(db, limit=CHANNEL_SUGGESTION_LIMIT)
    channel = event.channel
    if cleaned_channel_name:
        try:
            channel = ensure_channel(
                db, name=cleaned_channel_name, visibility=channel_visibility
            )
        except ValueError as exc:
            rsvps = list(event.rsvps)
            return templates.TemplateResponse(
                request,
                "event_admin.html",
                {
                    "request": request,
                    "event": event,
                    "rsvps": rsvps,
                    "rsvp_stats": rsvp_stats,
                    "available_yes_slots": available_yes_slots,
                    "event_is_full": event_is_full,
                    "approval_counts": approval_counts,
                    "admin_messages": admin_messages,
                    "rsvp_messages": rsvp_messages,
                    "public_channels": public_channels,
                    "admin_token": admin_token,
                    "message": _channel_error_message(str(exc)),
                    "message_class": "alert-danger",
                },
                status_code=400,
            )
    else:
        channel = None
    event = update_event(
        db,
        event,
        title=title,
        description=description,
        start_time=normalized_start,
        end_time=normalized_end,
        location=location,
        channel=channel,
        admin_approval_required=admin_approval_required,
        is_private=is_private,
        max_attendees=normalized_max,
        rsvps_closed=rsvps_closed,
        rsvp_close_at=normalized_close,
        update_rsvp_close_at=True,
    )
    rsvps = list(event.rsvps)
    rsvp_stats = _rsvp_stats(rsvps)
    available_yes_slots = _available_yes_slots(event)
    event_is_full = (
        available_yes_slots == 0 if available_yes_slots is not None else False
    )
    approval_counts = {
        "approved": sum(1 for r in rsvps if r.approval_status == "approved"),
        "pending": sum(1 for r in rsvps if r.approval_status == "pending"),
        "rejected": sum(1 for r in rsvps if r.approval_status == "rejected"),
    }
    admin_messages = _event_messages(
        event, visibilities={"public", "attendee", "admin"}
    )
    rsvp_messages = {
        rsvp.id: _rsvp_messages(rsvp, visibilities={"public", "attendee", "admin"})
        for rsvp in rsvps
    }
    return templates.TemplateResponse(
        request,
        "event_admin.html",
        {
            "request": request,
            "event": event,
            "rsvps": rsvps,
            "rsvp_stats": rsvp_stats,
            "available_yes_slots": available_yes_slots,
            "event_is_full": event_is_full,
            "approval_counts": approval_counts,
            "public_channels": public_channels,
            "admin_token": admin_token,
            "message": "Event updated",
            "message_class": "alert-success",
            "admin_messages": admin_messages,
            "rsvp_messages": rsvp_messages,
        },
    )


@app.post("/e/{event_id}/admin/{admin_token}/rsvp/{rsvp_id}/approve")
def approve_rsvp_admin(
    event_id: str,
    admin_token: str,
    rsvp_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    _apply_rsvp_status_change(db, event=event, rsvp=rsvp, new_status="approved")
    if _is_htmx(request):
        return _admin_rsvp_partial_response(
            request, event=event, admin_token=admin_token, rsvp=rsvp
        )
    params = urlencode({"message": "RSVP approved", "message_class": "alert-success"})
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
    )


@app.post("/e/{event_id}/admin/{admin_token}/rsvp/{rsvp_id}/reject")
def reject_rsvp_admin(
    event_id: str,
    admin_token: str,
    rsvp_id: str,
    request: Request,
    reason: str | None = Form(None),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    _apply_rsvp_status_change(
        db, event=event, rsvp=rsvp, new_status="rejected", reason=reason
    )
    if _is_htmx(request):
        return _admin_rsvp_partial_response(
            request, event=event, admin_token=admin_token, rsvp=rsvp
        )
    params = urlencode({"message": "RSVP rejected", "message_class": "alert-warning"})
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
    )


@app.post("/e/{event_id}/admin/{admin_token}/rsvp/{rsvp_id}/pending")
def pending_rsvp_admin(
    event_id: str,
    admin_token: str,
    rsvp_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    _apply_rsvp_status_change(db, event=event, rsvp=rsvp, new_status="pending")
    if _is_htmx(request):
        return _admin_rsvp_partial_response(
            request, event=event, admin_token=admin_token, rsvp=rsvp
        )
    params = urlencode(
        {"message": "RSVP set to pending", "message_class": "alert-info"}
    )
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
    )


@app.post("/e/{event_id}/admin/{admin_token}/rsvp/{rsvp_id}/force_accept")
def force_accept_rsvp_admin(
    event_id: str,
    admin_token: str,
    rsvp_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    try:
        forced = _enforce_attendance_capacity(
            event=event,
            new_status="yes",
            current_rsvp=rsvp,
            admin_token=admin_token,
            force_accept=True,
            new_guest_count=rsvp.guest_count,
        )
    except EventFullError:
        forced = False
    update_rsvp(
        db,
        rsvp=rsvp,
        name=rsvp.name,
        attendance_status="yes",
        approval_status=rsvp.approval_status,
        pronouns=rsvp.pronouns,
        guest_count=rsvp.guest_count,
        is_private=rsvp.is_private,
    )
    if forced:
        _log_force_accept(db, event=event, rsvp=rsvp)
    params = urlencode(
        {
            "message": "RSVP set to Yes (force accepted)",
            "message_class": "alert-success",
        }
    )
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
    )


@app.post("/e/{event_id}/admin/{admin_token}/rsvp/{rsvp_id}/note")
def add_admin_rsvp_note(
    event_id: str,
    admin_token: str,
    rsvp_id: str,
    request: Request,
    content: str = Form(...),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    message = _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=None,
        message_type="admin_note",
        visibility="admin",
        content=content,
    )
    if _is_htmx(request):
        return _admin_rsvp_partial_response(
            request, event=event, admin_token=admin_token, rsvp=rsvp
        )
    if not message:
        params = urlencode(
            {"message": "Note cannot be empty", "message_class": "alert-danger"}
        )
    else:
        params = urlencode({"message": "Note added", "message_class": "alert-success"})
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
    )


@app.post("/e/{event_id}/admin/{admin_token}/rsvp/{rsvp_id}/message")
def add_admin_rsvp_message(
    event_id: str,
    admin_token: str,
    rsvp_id: str,
    request: Request,
    content: str = Form(...),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    message = _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=None,
        message_type="admin_message",
        visibility="attendee",
        content=content,
    )
    if _is_htmx(request):
        return _admin_rsvp_partial_response(
            request, event=event, admin_token=admin_token, rsvp=rsvp
        )
    if not message:
        params = urlencode(
            {"message": "Message cannot be empty", "message_class": "alert-danger"}
        )
    else:
        params = urlencode(
            {"message": "Message sent", "message_class": "alert-success"}
        )
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
    )


@app.post("/e/{event_id}/admin/{admin_token}/messages")
def add_event_message_admin(
    event_id: str,
    admin_token: str,
    request: Request,
    content: str = Form(...),
    visibility: str = Form("public"),
    message_type: str = Form("event_update"),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    visibility = (visibility or "").lower()
    if visibility not in {"public", "admin"}:
        params = urlencode(
            {"message": "Invalid visibility", "message_class": "alert-danger"}
        )
        return RedirectResponse(
            url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
        )
    if message_type not in {"event_update", "event_internal"}:
        params = urlencode(
            {"message": "Invalid message type", "message_class": "alert-danger"}
        )
        return RedirectResponse(
            url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
        )
    message = _create_message_if_content(
        db,
        event=event,
        rsvp=None,
        author_id=None,
        message_type=message_type,
        visibility=visibility,
        content=content,
    )
    if not message:
        params = urlencode(
            {"message": "Message cannot be empty", "message_class": "alert-danger"}
        )
    else:
        params = urlencode(
            {"message": "Message posted", "message_class": "alert-success"}
        )
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{admin_token}?{params}", status_code=303
    )


@app.post("/e/{event_id}/admin/{admin_token}/delete")
def delete_event(
    event_id: str, admin_token: str, request: Request, db: Session = Depends(get_db)
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    db.delete(event)
    return templates.TemplateResponse(
        request, "event_deleted.html", {"request": request, "event_id": event_id}
    )


@app.get("/admin/{root_token}")
def root_admin(
    root_token: str,
    request: Request,
    page: int = Query(1, ge=1),
    channel_page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    _require_root_access(root_token)
    channels, channel_pagination = paginate_channels(
        db, query=None, page=channel_page, include_events=True
    )
    events, pagination = _paginate_events(db, page=page, query=None, include_rsvps=True)
    return templates.TemplateResponse(
        request,
        "root_admin.html",
        {
            "request": request,
            "events": events,
            "root_token": root_token,
            "channels": channels,
            "pagination": pagination,
            "channel_pagination": channel_pagination,
            "channel_query": "",
            "events_query": "",
        },
    )


@app.get("/admin/{root_token}/channels")
def root_admin_channels(
    root_token: str,
    request: Request,
    q: str | None = Query(None),
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    _require_root_access(root_token)
    channels, channel_pagination = paginate_channels(
        db, query=q, page=page, include_events=True
    )
    return templates.TemplateResponse(
        request,
        "partials/admin_channels_table.html",
        {
            "request": request,
            "channels": channels,
            "channel_query": q or "",
            "channel_pagination": channel_pagination,
            "root_token": root_token,
        },
    )


@app.get("/admin/{root_token}/events")
def root_admin_events(
    root_token: str,
    request: Request,
    page: int = Query(1, ge=1),
    q: str | None = Query(None),
    db: Session = Depends(get_db),
):
    _require_root_access(root_token)
    events, pagination = _paginate_events(
        db,
        page=page,
        query=q,
        include_rsvps=True,
    )
    return templates.TemplateResponse(
        request,
        "partials/admin_events_table.html",
        {
            "request": request,
            "events": events,
            "pagination": pagination,
            "root_token": root_token,
            "events_query": q or "",
        },
    )


@app.post("/admin/{root_token}/events/{event_id}/delete")
def root_delete_event(
    root_token: str, event_id: str, request: Request, db: Session = Depends(get_db)
):
    _require_root_access(root_token)
    event = _ensure_event(db, event_id)
    db.delete(event)
    return RedirectResponse(url=f"/admin/{root_token}", status_code=303)


def _render_channel_page(
    request: Request,
    db: Session,
    *,
    channel: Channel,
    page: int,
):
    touch_channel(channel)
    db.add(channel)
    include_private = channel.visibility == "private"
    events, pagination = _paginate_visible_events(
        db,
        include_private=include_private,
        channel_id=channel.id,
        page=page,
        per_page=EVENTS_PER_PAGE,
    )
    return templates.TemplateResponse(
        request,
        "channel.html",
        {
            "request": request,
            "channel": channel,
            "events": events,
            "pagination": pagination,
        },
    )


@app.get("/channel/{slug}")
def channel_page(
    slug: str,
    request: Request,
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    channel = get_channel_by_slug(db, slug)
    if not channel or channel.visibility != "public":
        raise HTTPException(status_code=404, detail="Channel not found")
    return _render_channel_page(request, db, channel=channel, page=page)


@app.get("/channel/p/{slug}")
def channel_page_private(
    slug: str,
    request: Request,
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    channel = get_channel_by_slug(db, slug)
    if not channel or channel.visibility != "private":
        raise HTTPException(status_code=404, detail="Channel not found")
    return _render_channel_page(request, db, channel=channel, page=page)


@app.get("/help")
@app.get("/help/events")
@app.get("/help/rsvp")
@app.get("/help/faq", response_class=HTMLResponse)
@app.get("/help/privacy", response_class=HTMLResponse)
@app.get("/help/features", response_class=HTMLResponse)
@app.get("/my-events")
def my_events_page(request: Request):
    return templates.TemplateResponse(
        request,
        "my_events.html",
        {
            "request": request,
            "view": "hosted",
            "page_title": "My Events",
            "empty_title": "No hosted events saved yet.",
            "empty_body": "Create an event or open its admin link on this device to store it.",
        },
    )


@app.get("/my-rsvps")
def my_rsvps_page(request: Request):
    return templates.TemplateResponse(
        request,
        "my_events.html",
        {
            "request": request,
            "view": "rsvps",
            "page_title": "My RSVPs",
            "empty_title": "No RSVP events saved yet.",
            "empty_body": "RSVP to an event on this device and it will appear here.",
        },
    )


def _paginate_discover_public_channels(
    db: Session,
    *,
    query: str | None,
    page: int,
    per_page: int,
    sort: str,
    has_upcoming: bool,
    invert: bool,
) -> tuple[list[Channel], dict]:
    clause = _channel_search_clause(query)
    filters = [Channel.visibility == "public"]
    if clause is not None:
        filters.append(clause)
    if has_upcoming:
        now = utcnow()
        upcoming_exists = (
            select(Event.id)
            .where(
                Event.channel_id == Channel.id,
                Event.is_private.is_(False),
                Event.start_time >= now,
            )
            .exists()
        )
        filters.append(upcoming_exists)

    count_stmt = select(func.count()).select_from(Channel)
    for condition in filters:
        count_stmt = count_stmt.where(condition)
    total_channels = db.scalar(count_stmt) or 0
    pagination = _build_pagination(
        page=page,
        per_page=per_page,
        total_events=total_channels,
        include_query=True,
        query=query,
    )
    page_number = pagination["page"]
    offset = (page_number - 1) * per_page if total_channels else 0

    sort_key = (sort or "rank").strip().lower()
    if sort_key == "name":
        order_by = [Channel.name.desc()] if invert else [Channel.name.asc()]
    elif sort_key == "recent":
        if invert:
            order_by = [
                Channel.last_used_at.asc(),
                Channel.score.asc(),
                Channel.name.asc(),
            ]
        else:
            order_by = [
                Channel.last_used_at.desc(),
                Channel.score.desc(),
                Channel.name.asc(),
            ]
    else:
        order_by = (
            [Channel.score.asc(), Channel.name.asc()]
            if invert
            else [Channel.score.desc(), Channel.name.asc()]
        )

    stmt = select(Channel)
    for condition in filters:
        stmt = stmt.where(condition)
    stmt = stmt.order_by(*order_by).offset(offset).limit(per_page)
    channels = db.scalars(stmt).all()
    return channels, pagination


@app.get("/channels/discover")
def discover_channels_page(
    request: Request,
    q: str | None = Query(None),
    sort: str = Query("rank"),
    invert: bool = Query(False),
    has_upcoming: bool = Query(False),
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    sort_key = (sort or "rank").strip().lower()
    if sort_key not in {"rank", "name", "recent"}:
        sort_key = "rank"
    channels, pagination = _paginate_discover_public_channels(
        db,
        query=q,
        page=page,
        per_page=DISCOVER_CHANNELS_PER_PAGE,
        sort=sort_key,
        has_upcoming=has_upcoming,
        invert=invert,
    )
    params: dict[str, str] = {}
    if q:
        params["q"] = q
    if sort_key != "rank":
        params["sort"] = sort_key
    if invert:
        params["invert"] = "1"
    if has_upcoming:
        params["has_upcoming"] = "1"
    pagination_query = urlencode(params)
    template_name = (
        "partials/discover_channels_results.html"
        if _is_htmx(request)
        else "discover_channels.html"
    )
    return templates.TemplateResponse(
        request,
        template_name,
        {
            "request": request,
            "channels": channels,
            "pagination": pagination,
            "query": q or "",
            "sort": sort_key,
            "invert": invert,
            "has_upcoming": has_upcoming,
            "pagination_query": pagination_query,
        },
    )


# -------- JSON API (v1) --------


@app.get("/api/v1/events")
def api_list_public_events(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(EVENTS_PER_PAGE, ge=1, le=50),
    start_after: str | None = Query(None),
    start_before: str | None = Query(None),
    db: Session = Depends(get_db),
):
    start_after_dt = _parse_iso_datetime_param("start_after", start_after)
    start_before_dt = _parse_iso_datetime_param("start_before", start_before)
    if start_after_dt and start_before_dt and start_before_dt < start_after_dt:
        raise HTTPException(
            status_code=400, detail="start_before must be after start_after"
        )
    now = utcnow()
    filters = _visible_event_filters(include_private=False, channel_id=None, now=now)
    if start_after_dt:
        filters.append(Event.start_time >= start_after_dt)
    if start_before_dt:
        filters.append(Event.start_time <= start_before_dt)
    events, pagination = paginate_events(
        db,
        filters=filters,
        order_by=Event.start_time.asc(),
        per_page=per_page,
        page=page,
    )
    token = _get_bearer_token(request)
    root_token = _fetch_root_token_in_session(db) if token else None
    rsvp = (
        _fetch_rsvp_by_token_in_session(db, token)
        if token and not (root_token and token == root_token)
        else None
    )
    payload_events = []
    for event in events:
        include_private = _can_view_private_channel(
            event, token=token, root_token=root_token
        )
        payload_events.append(
            _serialize_event(
                event,
                include_location=_can_view_event_location_list(
                    event, token=token, root_token=root_token, rsvp=rsvp
                ),
                include_private_channel=include_private,
                include_unapproved_yes_count=include_private,
            )
        )
    return {
        "events": payload_events,
        "pagination": pagination,
        "filters": {
            "start_after": start_after_dt.isoformat() if start_after_dt else None,
            "start_before": start_before_dt.isoformat() if start_before_dt else None,
        },
    }


@app.post("/api/v1/events", status_code=201)
def api_create_event(payload: EventCreatePayload, db: Session = Depends(get_db)):
    channel = None
    cleaned_channel = (payload.channel_name or "").strip()
    visibility = payload.channel_visibility or "public"
    if cleaned_channel:
        try:
            channel = ensure_channel(db, name=cleaned_channel, visibility=visibility)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=_channel_error_message(str(exc))
            ) from exc

    normalized_start, normalized_end = _normalize_event_times(
        start_time=payload.start_time,
        end_time=payload.end_time,
        timezone_offset_minutes=payload.timezone_offset_minutes,
    )
    normalized_close = _normalize_optional_close_time(
        payload.rsvp_close_at, payload.timezone_offset_minutes
    )
    if not normalized_start:
        raise HTTPException(status_code=400, detail="start_time is required")

    event = create_event(
        db,
        title=payload.title,
        description=payload.description,
        start_time=normalized_start,
        end_time=normalized_end,
        location=payload.location,
        channel=channel,
        is_private=payload.is_private,
        admin_approval_required=payload.admin_approval_required,
        max_attendees=payload.max_attendees,
        rsvps_closed=payload.rsvps_closed,
        rsvp_close_at=normalized_close,
    )
    return {
        "event": _serialize_event(
            event, include_admin_link=True, include_private_channel=True
        ),
        "admin_token": event.admin_token,
    }


@app.get("/api/v1/events/{event_id}")
def api_get_event(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    event.last_accessed = utcnow()
    db.add(event)
    public_rsvps = [
        r for r in event.rsvps if not r.is_private and r.approval_status == "approved"
    ]
    public_messages = _event_messages(event, visibilities={"public"})
    token = _get_bearer_token(request)
    root_token = _fetch_root_token_in_session(db) if token else None
    include_location = _can_view_event_location_api(
        db, event=event, token=token, root_token=root_token
    )
    include_private_channel = _can_view_private_channel(
        event, token=token, root_token=root_token
    )
    return {
        "event": _serialize_event(
            event,
            include_rsvps=public_rsvps,
            include_private_counts=True,
            include_admin_link=False,
            include_location=include_location,
            include_private_channel=include_private_channel,
            include_unapproved_yes_count=include_private_channel,
            include_messages=public_messages,
        )
    }


@app.get("/api/v1/events/{event_id}/event.ics")
def api_get_event_ics(event_id: str, request: Request, db: Session = Depends(get_db)):
    """Serve an event as a downloadable ICS file."""

    event = _ensure_event(db, event_id)
    event.last_accessed = utcnow()
    db.add(event)
    token = _get_bearer_token(request)
    root_token = _fetch_root_token_in_session(db) if token else None
    include_location = _can_view_event_location_api(
        db, event=event, token=token, root_token=root_token
    )
    ics_text = generate_ics(event, include_location=include_location)
    filename = f"event_{event_id}.ics"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=ics_text, media_type="text/calendar", headers=headers)


@app.patch("/api/v1/events/{event_id}")
def api_update_event(
    event_id: str,
    payload: EventUpdatePayload,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    data = payload.model_dump(exclude_unset=True)
    tz_offset = int(data.get("timezone_offset_minutes", 0) or 0)
    normalized_start, normalized_end = _normalize_event_times(
        start_time=data.get("start_time"),
        end_time=data.get("end_time"),
        timezone_offset_minutes=tz_offset,
    )
    update_close = False
    normalized_close = None
    if "rsvp_close_at" in data:
        normalized_close = _normalize_optional_close_time(
            data.get("rsvp_close_at"), tz_offset
        )
        update_close = True
    channel = event.channel
    if "channel_name" in data or "channel_visibility" in data:
        cleaned_channel = (data.get("channel_name") or "").strip()
        visibility = data.get("channel_visibility") or "public"
        if cleaned_channel:
            try:
                channel = ensure_channel(
                    db, name=cleaned_channel, visibility=visibility
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail=_channel_error_message(str(exc))
                ) from exc
        else:
            channel = None
    admin_approval_required = data.get(
        "admin_approval_required", event.admin_approval_required
    )
    event = update_event(
        db,
        event,
        title=data.get("title", event.title),
        description=data.get("description", event.description),
        start_time=normalized_start or event.start_time,
        end_time=normalized_end if "end_time" in data else event.end_time,
        location=data.get("location", event.location),
        channel=channel,
        admin_approval_required=admin_approval_required,
        is_private=data.get("is_private", event.is_private),
        max_attendees=data.get("max_attendees", event.max_attendees),
        rsvps_closed=data.get("rsvps_closed"),
        rsvp_close_at=normalized_close,
        update_rsvp_close_at=update_close,
    )
    return {
        "event": _serialize_event(
            event, include_admin_link=True, include_private_channel=True
        )
    }


@app.delete("/api/v1/events/{event_id}", status_code=204)
def api_delete_event(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    db.delete(event)
    return Response(status_code=204)


@app.get("/api/v1/events/{event_id}/rsvps")
def api_list_event_rsvps(
    event_id: str, request: Request, db: Session = Depends(get_db)
):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    rsvps = list(event.rsvps)
    return {
        "event": _serialize_event(
            event, include_admin_link=True, include_private_channel=True
        ),
        "rsvps": [
            _serialize_rsvp(r, include_token=True, include_internal_id=True)
            for r in rsvps
        ],
        "stats": _rsvp_stats(rsvps),
    }


def _update_status_and_serialize(
    db: Session,
    *,
    event: Event,
    rsvp: RSVP,
    status: str,
    reason: str | None,
):
    _apply_rsvp_status_change(
        db, event=event, rsvp=rsvp, new_status=status, reason=reason
    )
    visible_messages = _rsvp_messages(
        rsvp, visibilities={"public", "attendee", "admin"}
    )
    return {"rsvp": _serialize_rsvp(rsvp, include_messages=visible_messages)}


@app.post("/events/{event_id}/rsvps/{rsvp_id}/approve")
@app.post("/api/v1/events/{event_id}/rsvps/{rsvp_id}/approve")
def api_approve_rsvp(
    event_id: str,
    rsvp_id: str,
    request: Request,
    payload: StatusChangePayload | None = None,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    reason = payload.reason if payload else None
    return _update_status_and_serialize(
        db, event=event, rsvp=rsvp, status="approved", reason=reason
    )


@app.post("/events/{event_id}/rsvps/{rsvp_id}/reject")
@app.post("/api/v1/events/{event_id}/rsvps/{rsvp_id}/reject")
def api_reject_rsvp(
    event_id: str,
    rsvp_id: str,
    request: Request,
    payload: StatusChangePayload | None = None,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    reason = payload.reason if payload else None
    return _update_status_and_serialize(
        db, event=event, rsvp=rsvp, status="rejected", reason=reason
    )


@app.post("/events/{event_id}/rsvps/{rsvp_id}/pending")
@app.post("/api/v1/events/{event_id}/rsvps/{rsvp_id}/pending")
def api_pending_rsvp(
    event_id: str,
    rsvp_id: str,
    request: Request,
    payload: StatusChangePayload | None = None,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    reason = payload.reason if payload else None
    return _update_status_and_serialize(
        db, event=event, rsvp=rsvp, status="pending", reason=reason
    )


@app.post("/api/v1/events/{event_id}/rsvps/self/messages", status_code=201)
def api_create_attendee_message(
    event_id: str,
    payload: AttendeeMessagePayload,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    rsvp = _require_rsvp_from_header(event, request, db)
    message = _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=rsvp.id,
        message_type="user_note",
        visibility="attendee",
        content=payload.content,
    )
    if not message:
        raise HTTPException(status_code=400, detail="Content is required")
    attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
    return {
        "message": _serialize_message(message),
        "messages": [_serialize_message(m) for m in attendee_messages],
    }


@app.post("/events/{event_id}/messages", status_code=201)
def api_create_event_message(
    event_id: str,
    payload: MessageCreatePayload,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    visibility = (payload.visibility or "").lower()
    if visibility not in {"public", "admin"}:
        raise HTTPException(status_code=400, detail="Invalid visibility for message")
    if payload.message_type not in {"event_update", "event_internal"}:
        raise HTTPException(status_code=400, detail="Invalid message type for event")
    message = _create_message_if_content(
        db,
        event=event,
        rsvp=None,
        author_id=None,
        message_type=payload.message_type,
        visibility=visibility,
        content=payload.content,
    )
    if not message:
        raise HTTPException(status_code=400, detail="Content is required")
    return {"message": _serialize_message(message)}


@app.post("/events/{event_id}/rsvps/{rsvp_id}/messages", status_code=201)
@app.post("/api/v1/events/{event_id}/rsvps/{rsvp_id}/messages", status_code=201)
def api_create_rsvp_message(
    event_id: str,
    rsvp_id: str,
    payload: MessageCreatePayload,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    rsvp = _ensure_rsvp_by_id(db, event, rsvp_id)
    visibility = (payload.visibility or "").lower()
    if visibility not in {"admin", "attendee"}:
        raise HTTPException(
            status_code=400, detail="Invalid visibility for RSVP message"
        )
    allowed_types = {"admin_note", "rejection_reason", "user_note", "admin_message"}
    if payload.message_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid message type for RSVP")
    if payload.message_type == "admin_note" and visibility != "admin":
        raise HTTPException(
            status_code=400,
            detail="Admin notes must use admin visibility",
        )
    if payload.message_type == "admin_message" and visibility != "attendee":
        raise HTTPException(
            status_code=400,
            detail="Admin messages must be visible to attendees",
        )
    message = _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=None,
        message_type=payload.message_type,
        visibility=visibility,
        content=payload.content,
    )
    if not message:
        raise HTTPException(status_code=400, detail="Content is required")
    visible_messages = _rsvp_messages(
        rsvp, visibilities={"public", "attendee", "admin"}
    )
    return {
        "message": _serialize_message(message),
        "messages": [_serialize_message(m) for m in visible_messages],
    }


@app.post("/api/v1/events/{event_id}/rsvps", status_code=201)
def api_create_rsvp(
    event_id: str,
    request: Request,
    payload: RSVPCreatePayload,
    force_accept: bool = Query(False),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    normalized_status = _normalize_attendance_status(payload.attendance_status)
    guest_count = _clamp_guest_count(payload.guest_count)
    admin_token = _get_bearer_token(request) if force_accept else None
    if event.rsvps_closed:
        if force_accept:
            if not admin_token:
                raise HTTPException(status_code=403, detail="Invalid admin token")
            _require_admin_or_root(event, admin_token)
        else:
            return JSONResponse(RSVPS_CLOSED_ERROR, status_code=403)
    try:
        forced = _enforce_attendance_capacity(
            event=event,
            new_status=normalized_status,
            admin_token=admin_token,
            force_accept=force_accept,
            new_guest_count=guest_count,
        )
    except EventFullError:
        return JSONResponse(EVENT_FULL_ERROR, status_code=400)
    rsvp = create_rsvp(
        db,
        event=event,
        name=payload.name,
        attendance_status=normalized_status,
        pronouns=payload.pronouns,
        guest_count=guest_count,
        is_private=payload.is_private_rsvp,
    )
    if forced:
        _log_force_accept(db, event=event, rsvp=rsvp)
    _log_rsvp_created(db, event=event, rsvp=rsvp)
    _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=rsvp.id,
        message_type="user_note",
        visibility="attendee",
        content=payload.note,
    )
    return {
        "rsvp": _serialize_rsvp(rsvp, include_token=True),
        "links": {
            "self": f"/api/v1/events/{event.id}/rsvps/self",
            "html": f"/e/{event.id}/rsvp/{rsvp.rsvp_token}",
        },
    }


@app.get("/api/v1/events/{event_id}/rsvps/self")
def api_get_own_rsvp(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    rsvp = _require_rsvp_from_header(event, request, db)
    attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
    return {
        "rsvp": _serialize_rsvp(
            rsvp, include_token=True, include_messages=attendee_messages
        )
    }


@app.patch("/api/v1/events/{event_id}/rsvps/self")
def api_update_own_rsvp(
    event_id: str,
    payload: RSVPUpdatePayload,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    rsvp = _require_rsvp_from_header(event, request, db)
    data = payload.model_dump(exclude_unset=True)
    new_status = _normalize_attendance_status(
        data.get("attendance_status", rsvp.attendance_status)
    )
    guest_count = _clamp_guest_count(data.get("guest_count", rsvp.guest_count))
    try:
        _enforce_attendance_capacity(
            event=event,
            new_status=new_status,
            current_rsvp=rsvp,
            new_guest_count=guest_count,
            force_accept=False,
        )
    except EventFullError:
        return JSONResponse(EVENT_FULL_ERROR, status_code=400)
    update_rsvp(
        db,
        rsvp=rsvp,
        name=data.get("name", rsvp.name),
        attendance_status=new_status,
        pronouns=data.get("pronouns", rsvp.pronouns),
        guest_count=guest_count,
        is_private=data.get("is_private_rsvp", rsvp.is_private),
    )
    _create_message_if_content(
        db,
        event=event,
        rsvp=rsvp,
        author_id=rsvp.id,
        message_type="user_note",
        visibility="attendee",
        content=data.get("note"),
    )
    attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
    return {
        "rsvp": _serialize_rsvp(
            rsvp, include_token=True, include_messages=attendee_messages
        )
    }


@app.delete("/api/v1/events/{event_id}/rsvps/self", status_code=204)
def api_delete_own_rsvp(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    rsvp = _require_rsvp_from_header(event, request, db)
    db.delete(rsvp)
    return Response(status_code=204)


@app.delete("/api/v1/events/{event_id}/rsvps/{rsvp_token}", status_code=204)
def api_delete_rsvp_admin(
    event_id: str,
    rsvp_token: str,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_header(event, request)
    rsvp = _ensure_rsvp(db, event, rsvp_token)
    db.delete(rsvp)
    return Response(status_code=204)


@app.get("/api/v1/channels")
def api_public_channels(
    q: str | None = Query(None),
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    channels, pagination = _paginate_public_channels(db, query=q, page=page)
    return {
        "channels": [_serialize_channel(ch) for ch in channels],
        "pagination": pagination,
    }


@app.get("/api/v1/channels/{slug}")
def api_channel_detail(
    slug: str,
    request: Request,
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    channel = get_channel_by_slug(db, slug)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    touch_channel(channel)
    db.add(channel)
    include_private = channel.visibility == "private"
    events, pagination = _paginate_visible_events(
        db,
        include_private=include_private,
        channel_id=channel.id,
        page=page,
        per_page=EVENTS_PER_PAGE,
    )
    token = _get_bearer_token(request)
    root_token = _fetch_root_token_in_session(db) if token else None
    rsvp = (
        _fetch_rsvp_by_token_in_session(db, token)
        if token and not (root_token and token == root_token)
        else None
    )
    payload_events = []
    for event in events:
        include_private = _can_view_private_channel(
            event, token=token, root_token=root_token
        )
        payload_events.append(
            _serialize_event(
                event,
                include_location=_can_view_event_location_list(
                    event, token=token, root_token=root_token, rsvp=rsvp
                ),
                include_private_channel=include_private,
                include_unapproved_yes_count=include_private,
            )
        )
    return {
        "channel": _serialize_channel(channel),
        "events": payload_events,
        "pagination": pagination,
    }
