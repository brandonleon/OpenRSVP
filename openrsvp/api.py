"""FastAPI application for OpenRSVP."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .crud import (
    create_event,
    create_rsvp,
    ensure_channel,
    get_channel_by_slug,
    get_public_channels,
    touch_channel,
    update_event,
    update_rsvp,
)
from .database import SessionLocal
from .models import Channel, Event, RSVP
from .scheduler import start_scheduler, stop_scheduler
from .storage import fetch_root_token, init_db
from .utils import duration_between, get_repo_url, humanize_time, render_markdown

logger = logging.getLogger(__name__)

app = FastAPI(title="OpenRSVP", version="0.1")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.globals["app_version"] = app.version
templates.env.globals["repo_url"] = get_repo_url()
templates.env.filters["relative_time"] = humanize_time
templates.env.filters["duration"] = duration_between
templates.env.filters["markdown"] = render_markdown

ADMIN_EVENTS_PER_PAGE = 25
EVENTS_PER_PAGE = 10


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
    return "application/json" in accept and "text/html" not in accept


def _render_error(request: Request, status_code: int, message: str | None):
    context = {
        "request": request,
        "status_code": status_code,
        "message": None,
        "error_message": message or "Something went wrong.",
    }
    return templates.TemplateResponse("error.html", context, status_code=status_code)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Render HTTP errors as friendly pages unless JSON was requested."""
    if _wants_json(request):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    detail = exc.detail if isinstance(exc.detail, str) else "Something went wrong."
    return _render_error(request, exc.status_code, detail)


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


def _require_root_access(root_token: str) -> None:
    stored = fetch_root_token()
    if root_token != stored:
        raise HTTPException(status_code=403, detail="Forbidden")


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


def _paginate_events(db: Session, *, page: int, query: str | None):
    per_page = ADMIN_EVENTS_PER_PAGE
    clause = _event_search_clause(query)
    count_stmt = select(func.count()).select_from(Event)
    if clause is not None:
        count_stmt = count_stmt.where(clause)
    total_events = db.scalar(count_stmt) or 0
    total_pages = (
        max(1, (total_events + per_page - 1) // per_page) if total_events else 1
    )
    page = max(1, min(page, total_pages)) if total_events else 1
    offset = (page - 1) * per_page if total_events else 0
    stmt = select(Event).order_by(Event.created_at.desc())
    if clause is not None:
        stmt = stmt.where(clause)
    stmt = stmt.offset(offset).limit(per_page)
    events = db.scalars(stmt).all()
    for event in events:
        _ = list(event.rsvps)
    pagination = {
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_events": total_events,
        "has_prev": page > 1,
        "has_next": page < total_pages and total_events > 0,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < total_pages and total_events > 0 else None,
        "query": query or "",
    }
    return events, pagination


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
    now = datetime.utcnow()
    filters = _visible_event_filters(
        include_private=include_private,
        channel_id=channel_id,
        now=now,
    )
    count_stmt = select(func.count()).select_from(Event)
    for condition in filters:
        count_stmt = count_stmt.where(condition)
    total_events = db.scalar(count_stmt) or 0
    total_pages = (
        max(1, (total_events + per_page - 1) // per_page) if total_events else 1
    )
    page = max(1, min(page, total_pages)) if total_events else 1
    offset = (page - 1) * per_page if total_events else 0

    stmt = select(Event).order_by(Event.start_time.desc())
    for condition in filters:
        stmt = stmt.where(condition)
    stmt = stmt.offset(offset).limit(per_page)
    events = db.scalars(stmt).all()
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
    return events, pagination


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_scheduler()


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
        return RedirectResponse(url=f"/channel/{slug_channel.slug}", status_code=303)
    events, pagination = _paginate_visible_events(
        db,
        include_private=False,
        channel_id=None,
        page=page,
        per_page=EVENTS_PER_PAGE,
    )
    public_channels = get_public_channels(db)
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "events": events,
            "public_channels": public_channels,
            "pagination": pagination,
        },
    )


@app.get("/event/create")
def event_create_page(request: Request, db: Session = Depends(get_db)):
    public_channels = get_public_channels(db)
    return templates.TemplateResponse(
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
    end_time: str | None = Form(None),
    timezone_offset_minutes: int = Form(0),
    db: Session = Depends(get_db),
):
    channel = None
    cleaned_channel_name = channel_name.strip() if channel_name else ""
    if cleaned_channel_name:
        try:
            channel = ensure_channel(
                db, name=cleaned_channel_name, visibility=channel_visibility
            )
        except ValueError as exc:
            return templates.TemplateResponse(
                "event_create.html",
                {
                    "request": request,
                    "public_channels": get_public_channels(db),
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
                "event_create.html",
                {
                    "request": request,
                    "public_channels": get_public_channels(db),
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
    )
    return templates.TemplateResponse(
        "event_created.html",
        {
            "request": request,
            "event": event,
            "admin_link": f"/e/{event.id}/admin/{event.admin_token}",
        },
    )


@app.get("/e/{event_id}")
def event_page(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    event.last_accessed = datetime.utcnow()
    db.add(event)
    if event.channel:
        touch_channel(event.channel)
        db.add(event.channel)
    rsvps = list(event.rsvps)
    return templates.TemplateResponse(
        "event.html",
        {"request": request, "event": event, "rsvps": rsvps},
    )


@app.get("/e/{event_id}/rsvp")
def rsvp_form(event_id: str, request: Request, db: Session = Depends(get_db)):
    event = _ensure_event(db, event_id)
    return templates.TemplateResponse(
        "rsvp_form.html",
        {"request": request, "event": event},
    )


@app.post("/e/{event_id}/rsvp")
def create_rsvp_view(
    event_id: str,
    request: Request,
    name: str = Form(...),
    status: str = Form(...),
    pronouns: str | None = Form(None),
    guest_count: int = Form(0),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    guest_count = max(0, min(guest_count, 5))
    rsvp = create_rsvp(
        db,
        event=event,
        name=name,
        status=status,
        pronouns=pronouns,
        guest_count=guest_count,
        notes=notes,
    )
    return templates.TemplateResponse(
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
    return templates.TemplateResponse(
        "rsvp_edit.html",
        {"request": request, "event": event, "rsvp": rsvp},
    )


@app.post("/e/{event_id}/rsvp/{rsvp_token}")
def save_rsvp(
    event_id: str,
    rsvp_token: str,
    request: Request,
    name: str = Form(...),
    status: str = Form(...),
    pronouns: str | None = Form(None),
    guest_count: int = Form(0),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    rsvp = _ensure_rsvp(db, event, rsvp_token)
    guest_count = max(0, min(guest_count, 5))
    update_rsvp(
        db,
        rsvp=rsvp,
        name=name,
        status=status,
        pronouns=pronouns,
        guest_count=guest_count,
        notes=notes,
    )
    return templates.TemplateResponse(
        "rsvp_edit.html",
        {
            "request": request,
            "event": event,
            "rsvp": rsvp,
            "message": "RSVP updated",
            "message_class": "alert-success",
        },
    )


@app.get("/e/{event_id}/admin/{admin_token}")
def event_admin(
    event_id: str, admin_token: str, request: Request, db: Session = Depends(get_db)
):
    event = _ensure_event(db, event_id)
    if event.admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    rsvps = list(event.rsvps)
    return templates.TemplateResponse(
        "event_admin.html",
        {
            "request": request,
            "event": event,
            "rsvps": rsvps,
        },
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
    is_private: bool = Form(False),
    end_time: str | None = Form(None),
    timezone_offset_minutes: int = Form(0),
    db: Session = Depends(get_db),
):
    event = _ensure_event(db, event_id)
    if event.admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    parsed_start = _parse_datetime(start_time)
    normalized_start = _local_to_utc(parsed_start, timezone_offset_minutes)
    normalized_end = None
    if end_time:
        parsed_end = _parse_datetime(end_time)
        normalized_end = _local_to_utc(parsed_end, timezone_offset_minutes)
        if normalized_end <= normalized_start:
            rsvps = list(event.rsvps)
            return templates.TemplateResponse(
                "event_admin.html",
                {
                    "request": request,
                    "event": event,
                    "rsvps": rsvps,
                    "message": "End time must be after the start time.",
                    "message_class": "alert-danger",
                },
                status_code=400,
            )
    update_event(
        db,
        event,
        title=title,
        description=description,
        start_time=normalized_start,
        end_time=normalized_end,
        location=location,
        channel=event.channel,
        is_private=is_private,
    )
    rsvps = list(event.rsvps)
    return templates.TemplateResponse(
        "event_admin.html",
        {
            "request": request,
            "event": event,
            "rsvps": rsvps,
            "message": "Event updated",
            "message_class": "alert-success",
        },
    )


@app.post("/e/{event_id}/admin/{admin_token}/delete")
def delete_event(
    event_id: str, admin_token: str, request: Request, db: Session = Depends(get_db)
):
    event = _ensure_event(db, event_id)
    if event.admin_token != admin_token:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    db.delete(event)
    return templates.TemplateResponse(
        "event_deleted.html",
        {"request": request, "event_id": event_id},
    )


@app.get("/admin/{root_token}")
def root_admin(
    root_token: str,
    request: Request,
    page: int = Query(1, ge=1),
    db: Session = Depends(get_db),
):
    _require_root_access(root_token)
    channels = _fetch_channels(db, query=None)
    events, pagination = _paginate_events(db, page=page, query=None)
    return templates.TemplateResponse(
        "root_admin.html",
        {
            "request": request,
            "events": events,
            "root_token": root_token,
            "channels": channels,
            "pagination": pagination,
            "channel_query": "",
            "events_query": "",
        },
    )


@app.get("/admin/{root_token}/channels")
def root_admin_channels(
    root_token: str,
    request: Request,
    q: str | None = Query(None),
    db: Session = Depends(get_db),
):
    _require_root_access(root_token)
    channels = _fetch_channels(db, query=q)
    return templates.TemplateResponse(
        "partials/admin_channels_table.html",
        {
            "request": request,
            "channels": channels,
            "channel_query": q or "",
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
    events, pagination = _paginate_events(db, page=page, query=q)
    return templates.TemplateResponse(
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


@app.get("/channel/{slug}")
def channel_page(
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
    return templates.TemplateResponse(
        "channel.html",
        {
            "request": request,
            "channel": channel,
            "events": events,
            "pagination": pagination,
        },
    )


@app.get("/help")
def help_page(request: Request):
    return templates.TemplateResponse("help.html", {"request": request})
