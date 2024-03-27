from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from icecream import ic
from sqlmodel import Session, select
from starlette.templating import Jinja2Templates

from OpenRSVP import People
from OpenRSVP.utils import (format_timestamp, get_password_hash, get_session,
                            get_user_id_from_cookie, sanitize_markdown,
                            set_user_id_cookie)

router = APIRouter(prefix="/user")

# Templates
templates = Jinja2Templates(directory=f"{Path('templates').resolve()}")

# functions to be used in templates as filters
templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["sanitize_markdown"] = sanitize_markdown


def get_current_session(request: Request):
    if session_id := request.cookies.get("session_id"):
        return session_id
    else:
        raise HTTPException(
            status_code=303, headers={"Location": router.url_path_for("user_login")}
        )


@router.get("/me")
async def get_user_me(
    request: Request, current_user: str = Depends(get_current_session)
):
    return "Hello"


@router.get("/login", response_class=HTMLResponse, name="user_login")
async def login(response: Response, request: Request):
    return templates.TemplateResponse("user_login.html", {"request": request})


@router.post("/login", response_class=HTMLResponse, name="user_login")
async def post_login(
    response: Response,
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    # Check if the user exists
    user_ = session.exec(select(People).where(People.email == email)).first()
    if not user_:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return "User not found..."
    # Check if the password hash is correct
    if user_.pass_hash != ic(get_password_hash(user_.user_id, password, user_.salt)):
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return "Incorrect password..."

    response = RedirectResponse(url="/", status_code=303)
    # for debugging, construct the password hash
    return "Success!"


@router.get("/logout", response_class=HTMLResponse, name="user_logout")
async def logout(response: Response):
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("session_id")
    return response


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
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


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

    return RedirectResponse(url="/user", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login/{user_id}", response_class=HTMLResponse, name="user_login")
async def user_login(
    user_id: str,
    response: Response,
    session: Session = Depends(get_session),
):
    #  Check if user_id exists in the database
    if not session.get(People, user_id):
        response.status_code = 404
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    set_user_id_cookie(
        redirect_response := RedirectResponse(url="/", status_code=303), user_id
    )
    return redirect_response


@router.get("/create", response_class=HTMLResponse, name="user_create")
async def user_create(request: Request):
    return templates.TemplateResponse("user_create.html", {"request": request})
