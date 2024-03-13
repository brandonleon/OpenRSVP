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

from OpenRSVP.events import router as events_router


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
app.include_router(events_router)

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

    is_updated = False

    for f, v in {
        "display_name": display_name,
        "email": email,
        "cell_phone": cell_phone,
    }.items():
        if getattr(usr, f) != v:
            setattr(usr, f, v)
            is_updated = True

    usr.updated = datetime.now().timestamp() if is_updated else usr.updated
    # Format the display name and email
    usr.display_name = display_name.title() if usr.display_name else usr.display_name
    usr.email = email.lower() if usr.email else usr.email
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
