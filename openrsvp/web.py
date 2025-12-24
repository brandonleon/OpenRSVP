"""Web route handlers for OpenRSVP."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from .utils import get_repo_url

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _help_response(request: Request, template_name: str):
    return templates.TemplateResponse(
        request,
        template_name,
        {"request": request, "repo_url": get_repo_url()},
    )


def help_index(request: Request):
    """Render the help index page."""
    return _help_response(request, "help/index.html")


def help_events(request: Request):
    """Render the help events page."""
    return _help_response(request, "help/events.html")


def help_rsvp(request: Request):
    """Render the help RSVP page."""
    return _help_response(request, "help/rsvp.html")


def help_features(request: Request):
    """Render the help features page."""
    return _help_response(request, "help/features.html")


def help_privacy(request: Request):
    """Render the help privacy page."""
    return _help_response(request, "help/privacy.html")


def help_faq(request: Request):
    """Render the help FAQ page."""
    return _help_response(request, "help/faq.html")


def register_web_routes(app):
    """Register web routes on the FastAPI app."""
    app.get("/help", response_class=HTMLResponse)(help_index)
    app.get("/help/events", response_class=HTMLResponse)(help_events)
    app.get("/help/rsvp", response_class=HTMLResponse)(help_rsvp)
    app.get("/help/features", response_class=HTMLResponse)(help_features)
    app.get("/help/privacy", response_class=HTMLResponse)(help_privacy)
    app.get("/help/faq", response_class=HTMLResponse)(help_faq)
