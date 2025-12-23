"""HTMX Partial Route Handlers for OpenRSVP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from .api import templates
from .config import settings
from .crud import (
    get_admin_approval_counts,
    get_admin_events_list,
    get_channel_by_slug,
    get_public_channels,
    validate_admin_token,
)
from .database import SessionLocal
from .models import Channel, Event


def validate_event_admin_context(
    event_slug: Optional[str] = None,
    channel_slug: Optional[str] = None,
    admin_token: Optional[str] = None,
    db: Optional[Session] = None,
) -> dict:
    """
    Validate event or channel admin context.

    Args:
        event_slug: Slug of the event
        channel_slug: Slug of the channel
        admin_token: Admin access token
        db: Database session

    Returns:
        Dict with validated context or raises HTTPException
    """
    if db is None:
        db = SessionLocal()

    if not admin_token:
        raise HTTPException(status_code=401, detail="Admin token required")

    context = {"admin_token": admin_token}

    # Validate admin token and retrieve associated context
    try:
        if event_slug:
            # Implement get_event_by_admin_token if needed
            event = (
                db.query(Event)
                .filter(Event.admin_token == admin_token, Event.slug == event_slug)
                .first()
            )
            if not event:
                raise HTTPException(status_code=403, detail="Invalid admin token")
            context["event"] = event
        elif channel_slug:
            channel = get_channel_by_slug(db, channel_slug)
            if not channel or not validate_admin_token(db, admin_token, Channel):
                raise HTTPException(status_code=403, detail="Invalid admin token")
            context["channel"] = channel

        return context
    finally:
        db.close()


def admin_approval_counts(request: Request, event_slug: str, admin_token: str):
    """Render admin approval counts partial."""
    db = SessionLocal()
    try:
        context = validate_event_admin_context(
            event_slug=event_slug, admin_token=admin_token, db=db
        )

        counts = get_admin_approval_counts(db, context["event"].id)

        return templates.TemplateResponse(
            "partials/admin_approval_counts.html",
            {"request": request, "counts": counts, "event": context["event"]},
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching approval counts: {str(e)}"
        )
    finally:
        db.close()


def admin_events_table(
    request: Request,
    channel_slug: Optional[str] = None,
    admin_token: Optional[str] = None,
):
    """Render admin events table partial."""
    db = SessionLocal()
    try:
        context = validate_event_admin_context(
            channel_slug=channel_slug, admin_token=admin_token, db=db
        )

        channel = context.get("channel")
        events = get_admin_events_list(db, channel_id=channel.id if channel else None)

        return templates.TemplateResponse(
            "partials/admin_events_table.html",
            {"request": request, "events": events, "channel": channel},
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching events table: {str(e)}"
        )
    finally:
        db.close()


def discover_channels_results(request: Request, search_term: Optional[str] = None):
    """Render discover channels results partial."""
    db = SessionLocal()
    try:
        channels = get_public_channels(db, search_term)

        return templates.TemplateResponse(
            "partials/discover_channels_results.html",
            {"request": request, "channels": channels, "search_term": search_term},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error discovering channels: {str(e)}"
        )
    finally:
        db.close()


def register_partial_routes(app):
    """Register all partial routes on the FastAPI app."""
    # Admin partial routes with token validation
    app.get("/partials/admin/approval-counts", response_class=HTMLResponse)(
        admin_approval_counts
    )
    app.get("/partials/admin/events-table", response_class=HTMLResponse)(
        admin_events_table
    )

    # Public partial routes
    app.get("/partials/discover/channels", response_class=HTMLResponse)(
        discover_channels_results
    )
