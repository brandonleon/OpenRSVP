from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, func, select
from fastapi import APIRouter
from starlette.templating import Jinja2Templates

from OpenRSVP import Events, People
from OpenRSVP.utils import (
    get_user_id_from_cookie,
    format_timestamp,
    sanitize_markdown,
    set_user_id_cookie,
    pad_string,
    format_code_to_alphanumeric,
    get_or_set_user_id_cookie,
    get_session,
)

router = APIRouter(prefix="/events")

# Templates
templates = Jinja2Templates(directory=Path("templates"))

# functions to be used in templates as filters
templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["sanitize_markdown"] = sanitize_markdown


@router.get("/mine", response_class=HTMLResponse, name="events")
async def view_events(
    request: Request,
    session: Session = Depends(get_session),
    offset: int = 0,
    limit: int = 10,
    page: int = 1,
):
    user_id = get_user_id_from_cookie(request)
    usr = session.get(People, user_id) or {"user_id": user_id}
    statement = (
        select(Events)
        .where(Events.user_id == user_id)
        .order_by(Events.start_datetime)
        .offset(offset)
        .limit(limit)
    )
    events = session.exec(statement).all()
    # Get count of all events owned by the user

    # Pagination dictionary
    page = {
        "total_items": (
            total_items := session.exec(select(func.count(Events.secret_code))).one()
        ),
        "total_pages": int(total_items / limit) + 1,
        "current": page,
        "offset": offset,
        "limit": limit,
        "page": page,
    }

    return templates.TemplateResponse(
        "get_events.html",
        {
            "request": request,
            "events": events,
            "usr": usr,
            "page": page,
        },
    )


@router.get("/create", response_class=HTMLResponse)
async def event_root(
    request: Request,
    session: Session = Depends(get_session),
):
    template_response = templates.TemplateResponse(
        "get_event_create.html", {"request": request, "user_id": None}
    )
    user_id = get_user_id_from_cookie(request)
    if user_id is None or user_id == "None":
        user_id = str(uuid4())
        new_user = People(
            user_id=user_id,
            created=datetime.now().timestamp(),
            updated=datetime.now().timestamp(),
            last_login=datetime.now().timestamp(),
            role="user",
        )
        session.add(new_user)
        session.commit()
        session.refresh(new_user)
    template_response = set_user_id_cookie(template_response, user_id)
    template_response.context["user_id"] = user_id
    return template_response


@router.post("/create", response_class=HTMLResponse, name="create_event")
async def create_event(
    request: Request,
    secret_code: Optional[str] = Form(None),
    event_name: str = Form(...),
    event_details: Optional[str] = Form(None),
    start_date: str = Form(...),
    start_time: str = Form(...),
    url: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    end_date: Optional[str] = Form(None),
    end_time: Optional[str] = Form(None),
    session: Session = Depends(get_session),
):
    code = format_code_to_alphanumeric(secret_code)

    start_datetime = int(
        datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M").timestamp()
    )
    active = (
        1 if start_datetime > datetime.now().timestamp() else 0
    )  # Active if the event is in the future.

    if end_date is None or end_time is None:
        end_datetime = None
    else:
        end_datetime = int(
            datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M").timestamp()
        )

    user_id = get_user_id_from_cookie(request)

    padding = 0
    while True:
        try:
            code = pad_string(code, padding)
            new_event = Events(
                active=active,
                secret_code=code,
                name=event_name,
                url=url,
                location=location,
                user_id=user_id,
                event_details=event_details,
                created=datetime.now().timestamp(),
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )
            session.add(new_event)
            session.commit()
            session.refresh(new_event)
            break  # Exit the loop once the event is inserted.
        except IntegrityError:
            padding += 1
            session.rollback()
            continue  # Try again with a new padding.
    return RedirectResponse(
        router.url_path_for("event_by_id", event_id=code), status_code=303
    )


@router.get("/{event_id}", response_class=HTMLResponse, name="event_by_id")
async def view_event(
    request: Request,
    response: Response,
    event_id: str,
    session: Session = Depends(get_session),
):
    user_id = get_user_id_from_cookie(request)
    usr = session.get(People, user_id) or {"user_id": user_id}
    if event := session.get(Events, event_id):
        template = "get_event_id.html"
        context = {"request": request, "event": event, "usr": usr}

    else:
        response.status_code = 404
        template = "404.html"
        context = {"request": request, "usr": usr}
    template_response = templates.TemplateResponse(template, context)
    template_response = get_or_set_user_id_cookie(request, template_response)

    return template_response
