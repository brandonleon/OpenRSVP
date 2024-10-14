from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi import APIRouter
from starlette.templating import Jinja2Templates

from OpenRSVP.utils import (
    get_user_id_from_cookie,
    format_timestamp,
    sanitize_markdown,
    set_user_id_cookie,
)

router = APIRouter(prefix="/rsvp")

# Templates
templates = Jinja2Templates(directory=Path("templates"))

# functions to be used in templates as filters
templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["sanitize_markdown"] = sanitize_markdown


@router.get("/{event_id}/{user_id}", response_class=HTMLResponse, name="rsvp")
async def rsvp(
    request: Request,
    event_id: str,
    user_id: str,
):
    cookie_user_id = get_user_id_from_cookie(request)
    template_response = templates.TemplateResponse(
        "get_rsvp.html",
        {"request": request, "event_id": event_id, "user_id": user_id},
    )

    if not cookie_user_id:
        # If not, set the user_id cookie to the user_id from the URL
        template_response = set_user_id_cookie(template_response, user_id)
        template_response.context["user_id"] = user_id

    return template_response
