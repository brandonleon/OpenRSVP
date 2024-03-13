from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, func, select
from starlette.staticfiles import StaticFiles

from OpenRSVP import (
    RSVP,
    Events,
    People,
    create_tables,
    engine,
    get_user_id_from_cookie,
    set_user_id_cookie,
)
from OpenRSVP.utils import (
    format_code_to_alphanumeric,
    format_timestamp,
    get_or_set_user_id_cookie,
    get_user_id_from_cookie,
    pad_string,
    sanitize_markdown,
)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Does the database file exist?
    if not Path("events.db").exists():
        create_tables()

    # Static files (CSS, JS, etc.)
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)

    # Static files (CSS, JS, etc.)
    app_.mount("/static", StaticFiles(directory=static_dir), name="static")
    yield
    print("Shutting down...")


# Initialize the FastAPI app
app = FastAPI(lifespan=lifespan)

# Templates
templates = Jinja2Templates(directory=Path("templates"))

# functions to be used in templates as filters
templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["sanitize_markdown"] = sanitize_markdown


def get_session():
    with Session(engine) as session:
        yield session


@app.get("/", response_class=HTMLResponse, name="root")
async def root(request: Request):
    user_id = get_user_id_from_cookie(request)
    return templates.TemplateResponse(
        "get_index.html", {"request": request, "user_id": user_id}
    )


@app.get("/event/create", response_class=HTMLResponse)
async def event_root(
    request: Request,
    session: Session = Depends(get_session),
):
    template_response = templates.TemplateResponse(
        "get_event.html", {"request": request, "user_id": None}
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


@app.post("/event/create", response_class=HTMLResponse, name="create_event")
async def create_event(
    request: Request,
    secret_code: Optional[str] = Form(None),
    event_name: str = Form(...),
    event_details: Optional[str] = Form(None),
    start_date: str = Form(...),
    start_time: str = Form(...),
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
            code = pad_string(secret_code, padding)
            new_event = Events(
                active=active,
                secret_code=code,
                name=event_name,
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

    return RedirectResponse(url=f"/event/{code}", status_code=303)


@app.get("/event/{event_id}", response_class=HTMLResponse, name="event_id")
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


@app.get("/events", response_class=HTMLResponse, name="events")
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
        {"request": request, "events": events, "usr": usr, "page": page},
    )


@app.get("/rsvp/{event_id}/{user_id}", response_class=HTMLResponse, name="rsvp")
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


@app.get("/user", response_class=HTMLResponse, name="user")
async def user(
    request: Request,
    session: Session = Depends(get_session),
):
    user_id = get_user_id_from_cookie(request)
    if usr := session.get(People, user_id):
        return templates.TemplateResponse(
            "get_user.html", {"request": request, "usr": usr}
        )
    else:
        return RedirectResponse(url="/", status_code=303)


@app.post("/user", response_class=HTMLResponse, name="user")
async def update_user(
    request: Request,
    display_name: str = Form(None),
    email: str = Form(None),
    cell_phone: str = Form(None),
    session: Session = Depends(get_session),
):
    # Update just the supplied fields
    user_id = get_user_id_from_cookie(request)
    usr = session.get(People, user_id)

    # Strip all characters except digits from the cell_phone, if present
    cell_phone = "".join([c for c in cell_phone if c.isdigit()]) if cell_phone else None

    is_updated = False

    for f, v in {
        "display_name": display_name,
        "email": email,
        "cell_phone": cell_phone,
    }.items():
        if v and getattr(usr, f) != v:
            setattr(usr, f, v)
            is_updated = True

    usr.updated = datetime.now().timestamp() if is_updated else usr.updated
    session.add(usr)
    session.commit()

    return RedirectResponse(url="/user", status_code=303)


@app.get("/user/login/{user_id}", response_class=HTMLResponse, name="user_login")
async def user_login(
    user_id: str,
    response: Response,
    session: Session = Depends(get_session),
):
    #  Check if user_id exists in the database
    if not session.get(People, user_id):
        response.status_code = 404
        return RedirectResponse(url="/", status_code=303)

    set_user_id_cookie(
        redirect_response := RedirectResponse(url="/", status_code=303), user_id
    )
    return redirect_response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8765)
