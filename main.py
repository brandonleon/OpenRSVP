from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from OpenRSVP.database import fetch_event, init_db, insert_event, fetch_config
from OpenRSVP.utils import code_format, format_timestamp, pad_string, sanitize_markdown

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
async def root(request: Request):
    user_expire_time = int(fetch_config("user_expire_time"))
    response = templates.TemplateResponse("get_index.html", {"request": request})
    if not (cookie := request.cookies.get("user_id")):
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


@app.get("/event", response_class=HTMLResponse)
async def event_root(request: Request):
    user_expire_time = fetch_config("user_expire_time")
    response = templates.TemplateResponse("get_event.html", {"request": request})
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


@app.post("/event", response_class=HTMLResponse, name="event")
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
    code = code_format(secret_code)

    start_datetime = int(
        datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M").timestamp()
    )
    if end_date is None or end_time is None:
        end_datetime = None
    else:
        end_datetime = int(
            datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M").timestamp()
        )

    if not (user_id := request.cookies.get("user_id")):
        user_id = str(uuid4())
        response.set_cookie(key="user_id", value=user_id, httponly=True)

    if insert_event(
        code, user_id, event_name, event_details, start_datetime, end_datetime
    ) == (
        False,
        "UNIQUE constraint failed: events.secret_code",
    ):
        padding = 2
        while True:
            padded_code = pad_string(code, padding)
            result = insert_event(padded_code, user_id, start_datetime, end_datetime)
            if result != (False, "UNIQUE constraint failed: events.secret_code"):
                break
            padding += 1

    return RedirectResponse(url=f"/event/{code}", status_code=303)


@app.get("/event/{event_id}", response_class=HTMLResponse)
async def view_event(request: Request, event_id: str):
    return templates.TemplateResponse(
        "get_event_id.html", {"request": request, "event": fetch_event(event_id)}
    )


@app.post("/rsvp", response_class=HTMLResponse)
async def rsvp(request: Request):
    return templates.TemplateResponse("rsvp.html", {"request": request})


@app.get("/user", response_class=HTMLResponse, name="user")
async def user(request: Request):
    # Get user_id from cookie
    try:
        user_id = request.cookies.get("user_id")
    except Exception as e:
        print(e)
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        "get_user.html", {"request": request, "user": user_id}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
