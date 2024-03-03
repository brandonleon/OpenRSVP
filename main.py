from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request, Depends
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from icecream import ic
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, Table, insert, select
from starlette.staticfiles import StaticFiles

from OpenRSVP import Config, create_tables, engine, Events, People
from OpenRSVP.database import fetch_user
from OpenRSVP.utils import (
    format_code_to_alphanumeric,
    format_timestamp,
    get_or_set_user_id_cookie,
    pad_string,
    sanitize_markdown,
    get_user_id_from_cookie,
)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    session = Session(engine)

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
async def root(request: Request, response: Response):
    template_response = templates.TemplateResponse(
        "get_index.html", {"request": request}
    )
    template_response = get_or_set_user_id_cookie(request, template_response)

    return template_response


@app.get("/event/create", response_class=HTMLResponse)
async def event_root(request: Request):
    template_response = templates.TemplateResponse(
        "event_create.html", {"request": request}
    )
    template_response = get_or_set_user_id_cookie(request, template_response)
    return template_response


@app.post("/event/create", response_class=HTMLResponse, name="create_event")
async def create_event(
    request: Request,
    response: Response,
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
                active=1,  # TODO: Active if the event is in the future.
                secret_code=code,
                name=event_name,
                user_id=user_id,
                event_details=event_details,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )
            session.add(new_event)
            session.commit()
            session.refresh(new_event)
            break  # Exit the loop once the event is inserted.
        except IntegrityError as e:
            padding += 1
            session.rollback()
            continue  # Try again with a new padding.

    return RedirectResponse(url=f"/event/{code}", status_code=303)


@app.get("/event/{event_id}", response_class=HTMLResponse)
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


@app.post("/rsvp", response_class=HTMLResponse)
async def rsvp(request: Request):
    return templates.TemplateResponse("rsvp.html", {"request": request})


@app.get("/user", response_class=HTMLResponse, name="user")
async def user(
    request: Request,
    session: Session = Depends(get_session),
):
    user_id = get_user_id_from_cookie(request)
    usr = session.get(People, user_id) or {"user_id": user_id}
    return templates.TemplateResponse("get_user.html", {"request": request, "usr": usr})


@app.post("/user", response_class=HTMLResponse, name="user")
async def update_user(
    request: Request,
    response: Response,
    display_name: str = Form(None),
    email: str = Form(None),
    cell_phone: str = Form(None),
    session: Session = Depends(get_session),
):
    # Update just the supplied fields
    user_id = get_user_id_from_cookie(request)
    usr = session.get(People, user_id)

    # Strip all characters except digits from the cell_phone
    cell_phone = "".join([c for c in cell_phone if c.isdigit()])

    for f, v in {
        "display_name": display_name,
        "email": email,
        "cell_phone": cell_phone,
    }.items():
        if v:
            setattr(usr, f, v)
    session.add(usr)
    session.commit()

    return RedirectResponse(url="/user", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8765)
