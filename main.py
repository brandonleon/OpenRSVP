from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

from OpenRSVP import events

app = FastAPI()

# Static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/event", response_class=HTMLResponse)
async def event_root(request: Request):
    return templates.TemplateResponse("new_event.html", {"request": request})


@app.post("/event", response_class=HTMLResponse)
async def create_event(event_name: str = Form(...), event_date: str = Form(...),
                       start_time: str = Form(...), end_time: str = Form(...), secret_code: str = Form(...)):
    print("test")
    events.create_event(event_name, event_date, start_time, end_time, secret_code)

    return RedirectResponse(url="/event/{event_id}")


@app.get("/event/{event_id}", response_class=HTMLResponse)
async def view_event(request: Request, event_id: str):
    print(request, event_id)
    return templates.TemplateResponse("event.html", {"request": request, "event_id": event_id})


@app.post("/rsvp", response_class=HTMLResponse)
async def rsvp(request: Request):
    return templates.TemplateResponse("rsvp.html", {"request": request})
