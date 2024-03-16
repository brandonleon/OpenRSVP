from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

from OpenRSVP import (
    create_tables,
)
from OpenRSVP.utils import (
    format_timestamp,
    get_user_id_from_cookie,
    sanitize_markdown,
)

from OpenRSVP.events import router as events_router
from OpenRSVP.user import router as user_router
from OpenRSVP.rsvp import router as rsvp_router


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
app.include_router(user_router)
app.include_router(rsvp_router)

# Templates
templates = Jinja2Templates(directory=Path("templates"))

# functions to be used in templates as filters
templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["sanitize_markdown"] = sanitize_markdown


@app.get("/", response_class=HTMLResponse, name="root")
async def root(request: Request):
    user_id = get_user_id_from_cookie(request)
    return templates.TemplateResponse(
        "get_index.html", {"request": request, "user_id": user_id}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8765)
