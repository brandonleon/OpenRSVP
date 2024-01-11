from typing import Annotated, Dict, Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from OpenRSVP.database import init_db, insert_event
from OpenRSVP.utils import code_format
from OpenRSVP.utils import pad_string
from datetime import datetime

from icecream import ic

# Initialize the database if it doesn't exist
init_db()

app = FastAPI()

# Static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("get_index.html", {"request": request})


@app.get("/event", response_class=HTMLResponse)
async def event_root(request: Request):
    return templates.TemplateResponse("get_event.html", {"request": request})


@app.post("/event", response_class=HTMLResponse)
async def create_event(
    secret_code: Optional[str] = Form(None),
    event_name: str = Form(...),
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

    if insert_event(code, event_name, start_datetime, end_datetime) == (
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

    return RedirectResponse(url=f"/event/{code}")


@app.get("/event/{event_id}", response_class=HTMLResponse)
async def view_event(request: Request, event_id: str):
    print(request, event_id)
    return templates.TemplateResponse(
        "event.html", {"request": request, "event_id": event_id}
    )


@app.post("/rsvp", response_class=HTMLResponse)
async def rsvp(request: Request):
    return templates.TemplateResponse("rsvp.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
