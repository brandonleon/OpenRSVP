from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

app = FastAPI()

# Static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Templates
templates = Jinja2Templates(directory="templates")


# Routes
@app.get("/event", response_class=HTMLResponse)
async def event_root(request: Request):
    return templates.TemplateResponse("event.html", {"request": request})
