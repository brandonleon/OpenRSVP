"""FastAPI application for OpenRSVP."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import tomllib

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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
)
from .database import SessionLocal
from .models import Channel, Event, Message, RSVP
from .scheduler import start_scheduler, stop_scheduler
from .storage import fetch_root_token, init_db
from .utils import (
    duration_between,
    get_repo_url,
    humanize_time,
    render_markdown,
    utcnow,
)

# Use uvicorn's error logger so messages get the level prefix in the default log
# format (needed for downstream filtering like Loki).
logger = logging.getLogger("uvicorn.error")


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
templates.env.globals["app_version"] = APP_VERSION
templates.env.globals["repo_url"] = get_repo_url()
templates.env.filters["relative_time"] = humanize_time
templates.env.filters["duration"] = duration_between
templates.env.filters["markdown"] = render_markdown

ADMIN_EVENTS_PER_PAGE = 25
EVENTS_PER_PAGE = 10
RECENT_EVENTS_LIMIT = 5
CHANNEL_SUGGESTION_LIMIT = 12


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


@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_touch_icon():
    return FileResponse(static_dir / "apple-touch-icon.png")


@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def apple_touch_icon_precomposed():
    return FileResponse(static_dir / "apple-touch-icon-precomposed.png")


def _ensure_event(db: Session, event_id: str) -> Event:
    event = db.get(Event, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
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
    include_messages: list[Message] | None = None,
):
    payload = {
        "id": event.id,
        "title": event.title,
        "description": event.description,
        "start_time": event.start_time.isoformat(),
        "end_time": event.end_time.isoformat() if event.end_time else None,
        "location": event.location,
        "is_private": event.is_private,
        "admin_approval_required": event.admin_approval_required,
        "score": event.score,
        "created_at": event.created_at.isoformat(),
        "last_modified": event.last_modified.isoformat(),
        "last_accessed": event.last_accessed.isoformat(),
        "links": {
            "public": f"/e/{event.id}",
        },
    }
    if event.channel:
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
    return db.scalars(stmt).all()


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
    public_channels = get_public_channels(db)
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
    end_time: str | None = Form(None),
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
    message = request.query_params.get("message")
    message_class = request.query_params.get("message_class")
    public_messages = _event_messages(event, visibilities={"public"})
    return templates.TemplateResponse(
        request,
        "event.html",
        {
            "request": request,
            "event": event,
            "rsvps": rsvps,
            "private_rsvp_count": private_rsvp_count,
            "public_party_size": public_party_size,
            "private_party_size": private_party_size,
            "message": message,
            "message_class": message_class,
            "public_messages": public_messages,
        },
    )


@app.get("/e/{event_id}/rsvp")
def rsvp_form(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    return templates.TemplateResponse(
        request, "rsvp_form.html", {"request": request, "event": event}
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
    guest_count = max(0, min(guest_count, 5))
    rsvp = create_rsvp(
        db,
        event=event,
        name=name,
        attendance_status=attendance_status,
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
    attendee_messages = _rsvp_messages(rsvp, visibilities={"attendee"})
    return templates.TemplateResponse(
        request,
        "rsvp_edit.html",
        {
            "request": request,
            "event": event,
            "rsvp": rsvp,
            "attendee_messages": attendee_messages,
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
    guest_count = max(0, min(guest_count, 5))
    update_rsvp(
        db,
        rsvp=rsvp,
        name=name,
        attendance_status=attendance_status,
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
    return templates.TemplateResponse(
        request,
        "rsvp_edit.html",
        {
            "request": request,
            "event": event,
            "rsvp": rsvp,
            "attendee_messages": attendee_messages,
            "message": "RSVP updated",
            "message_class": "alert-success",
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
    approval_counts = {
        "approved": sum(1 for r in rsvps if r.approval_status == "approved"),
        "pending": sum(1 for r in rsvps if r.approval_status == "pending"),
        "rejected": sum(1 for r in rsvps if r.approval_status == "rejected"),
    }
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
    return templates.TemplateResponse(
        request,
        "event_admin.html",
        {
            "request": request,
            "event": event,
            "rsvps": rsvps,
            "rsvp_stats": rsvp_stats,
            "approval_counts": approval_counts,
            "public_channels": public_channels,
            "admin_token": admin_token,
            "message": message,
            "message_class": message_class,
            "admin_messages": admin_messages,
            "rsvp_messages": rsvp_messages,
        },
    )


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
    end_time: str | None = Form(None),
    timezone_offset_minutes: int = Form(0),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvps = list(event.rsvps)
    rsvp_stats = _rsvp_stats(rsvps)
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
    )
    rsvps = list(event.rsvps)
    rsvp_stats = _rsvp_stats(rsvps)
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
            "approval_counts": approval_counts,
            "public_channels": public_channels,
            "admin_token": admin_token,
            "message": "Event updated",
            "message_class": "alert-success",
            "admin_messages": admin_messages,
            "rsvp_messages": rsvp_messages,
        },
    )


@app.post("/e/{event_id}/admin/{admin_token}/rsvp/{rsvp_token}/delete")
def delete_rsvp_admin(
    event_id: str,
    admin_token: str,
    rsvp_token: str,
    request: Request,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    _require_admin_or_root(event, admin_token)
    rsvp = _ensure_rsvp(db, event, rsvp_token)
    db.delete(rsvp)
    return RedirectResponse(
        url=f"/e/{event.id}/admin/{event.admin_token}", status_code=303
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
    params = urlencode(
        {"message": "RSVP set to pending", "message_class": "alert-info"}
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
    if not message:
        params = urlencode(
            {"message": "Note cannot be empty", "message_class": "alert-danger"}
        )
    else:
        params = urlencode({"message": "Note added", "message_class": "alert-success"})
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
def help_page(request: Request):
    return templates.TemplateResponse(request, "help.html", {"request": request})


# -------- JSON API (v1) --------


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
    )
    return {
        "event": _serialize_event(event, include_admin_link=True),
        "admin_token": event.admin_token,
    }


@app.get("/api/v1/events/{event_id}")
def api_get_event(event_id: str, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    event.last_accessed = utcnow()
    db.add(event)
    public_rsvps = [
        r for r in event.rsvps if not r.is_private and r.approval_status == "approved"
    ]
    public_messages = _event_messages(event, visibilities={"public"})
    return {
        "event": _serialize_event(
            event,
            include_rsvps=public_rsvps,
            include_private_counts=True,
            include_admin_link=False,
            include_messages=public_messages,
        )
    }


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
    )
    return {"event": _serialize_event(event, include_admin_link=True)}


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
        "event": _serialize_event(event, include_admin_link=True),
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
    if payload.message_type not in {"admin_note", "rejection_reason", "user_note"}:
        raise HTTPException(status_code=400, detail="Invalid message type for RSVP")
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
    payload: RSVPCreatePayload,
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    rsvp = create_rsvp(
        db,
        event=event,
        name=payload.name,
        attendance_status=payload.attendance_status,
        pronouns=payload.pronouns,
        guest_count=_clamp_guest_count(payload.guest_count),
        is_private=payload.is_private_rsvp,
    )
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
    update_rsvp(
        db,
        rsvp=rsvp,
        name=data.get("name", rsvp.name),
        attendance_status=data.get("attendance_status", rsvp.attendance_status),
        pronouns=data.get("pronouns", rsvp.pronouns),
        guest_count=_clamp_guest_count(data.get("guest_count", rsvp.guest_count)),
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
    return {
        "channel": _serialize_channel(channel),
        "events": [_serialize_event(event) for event in events],
        "pagination": pagination,
    }
