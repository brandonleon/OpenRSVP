from datetime import datetime
from typing import Annotated
from typing import Dict
from typing import Optional

from fastapi import FastAPI
from fastapi import Form
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from markdown import markdown
from starlette.staticfiles import StaticFiles

from OpenRSVP.database import init_db
from OpenRSVP.database import insert_event
from OpenRSVP.database import fetch_event
from OpenRSVP.utils import code_format
from OpenRSVP.utils import pad_string
from OpenRSVP.utils import format_timestamp

# Initialize the database if it doesn't exist
init_db()

# Initialize the FastAPI app
app = FastAPI()

# Static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["markdown"] = markdown


@app.get("/", response_class=HTMLResponse, name="root")
async def root(request: Request):
    return templates.TemplateResponse("get_index.html", {"request": request})


@app.get("/event", response_class=HTMLResponse)
async def event_root(request: Request):
    return templates.TemplateResponse("get_event.html", {"request": request})


@app.post("/event", response_class=HTMLResponse)
async def create_event(
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

    if insert_event(code, event_name, event_details, start_datetime, end_datetime) == (
        False,
        "UNIQUE constraint failed: events.secret_code",
    ):
        padding = 2
        while True:
            padded_code = pad_string(code, padding)
            result = insert_event(padded_code, event_name, start_datetime, end_datetime)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
