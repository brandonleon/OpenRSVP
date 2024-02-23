from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from OpenRSVP.database import (
    fetch_event,
    fetch_user,
    init_db,
    insert_event,
    fetch_config,
)
from OpenRSVP.utils import (
    format_code_to_alphanumeric,
    format_timestamp,
    pad_string,
    sanitize_markdown,
    get_or_set_user_id_cookie,
)

# Initialize the database if it doesn't exist
init_db()

# Initialize the FastAPI app
app = FastAPI()

# Static files (CSS, JS, etc.)
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates = Jinja2Templates(directory=Path("templates"))

# functions to be used in templates as filters
templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["sanitize_markdown"] = sanitize_markdown


@app.get("/", response_class=HTMLResponse, name="root")
async def root(request: Request, response: Response):
    get_or_set_user_id_cookie(request, response)
    return templates.TemplateResponse("get_index.html", {"request": request})


@app.get("/event/create", response_class=HTMLResponse)
async def event_root(request: Request):
    user_expire_time = fetch_config("user_expire_time")
    response = templates.TemplateResponse("event_create.html", {"request": request})
    if cookie := request.cookies.get("user_id"):
        response.set_cookie(
            key="user_id",
            value=cookie,
            httponly=True,
            expires=user_expire_time,
        )
    else:
        response.set_cookie(
            key="user_id",
            value=str(uuid4()),
            httponly=True,
            expires=user_expire_time,
        )
    return response


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

    user_id = get_or_set_user_id_cookie(request, response)

    if insert_event(
        code, event_name, user_id, event_details, str(start_datetime), str(end_datetime)
    ) == (
        False,
        "UNIQUE constraint failed: events.secret_code",
    ):
        padding = 1
        starting_code = code
        while True:
            code = pad_string(starting_code, padding)
            result = insert_event(
                code,
                event_name,
                user_id,
                event_details,
                str(start_datetime),
                str(end_datetime),
            )
            if result != (False, "UNIQUE constraint failed: events.secret_code"):
                break
            padding += 1

    return RedirectResponse(url=f"/event/{code}", status_code=303)


@app.get("/event/{event_id}", response_class=HTMLResponse)
async def view_event(request: Request, response: Response, event_id: str):
    user_id = get_or_set_user_id_cookie(request, response)
    usr = fetch_user(user_id) or {}
    if not usr:
        usr["user_id"] = user_id
    return templates.TemplateResponse(
        "get_event_id.html",
        {
            "request": request,
            "usr": usr,
            "event": fetch_event(event_id),
        },
    )


@app.post("/rsvp", response_class=HTMLResponse)
async def rsvp(request: Request):
    return templates.TemplateResponse("rsvp.html", {"request": request})


@app.get("/user", response_class=HTMLResponse, name="user")
async def user(request: Request):
    # Get user_id from cookie
    user_id = get_or_set_user_id_cookie(request, Response())
    usr = fetch_user(user_id)
    return templates.TemplateResponse("get_user.html", {"request": request, "usr": usr})


@app.post("/user", response_class=HTMLResponse, name="user")
async def update_user(
    request: Request,
    response: Response,
):
    return {"cookie": request.cookies.get("user_id")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8765)
