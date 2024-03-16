from datetime import datetime
from pathlib import Path

from fastapi import Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlmodel import Session
from fastapi import APIRouter
from starlette.templating import Jinja2Templates

from OpenRSVP import People
from OpenRSVP.utils import (
    get_user_id_from_cookie,
    format_timestamp,
    sanitize_markdown,
    set_user_id_cookie,
    get_session,
)

router = APIRouter(prefix="/user")

# Templates
templates = Jinja2Templates(directory=Path("templates"))

# functions to be used in templates as filters
templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["sanitize_markdown"] = sanitize_markdown


@router.get("/", response_class=HTMLResponse, name="user")
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


@router.post("/", response_class=HTMLResponse, name="user")
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


@router.get("/login/{user_id}", response_class=HTMLResponse, name="user_login")
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
