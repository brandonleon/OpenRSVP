from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlmodel import Session, select
from starlette.templating import Jinja2Templates

from OpenRSVP import Config, People, UserSession, engine
from OpenRSVP.utils import (
    format_timestamp,
    get_password_hash,
    get_session,
    get_user_id_from_cookie,
    sanitize_markdown,
    set_user_id_cookie,
)

router = APIRouter(prefix="/user")

# Templates
templates = Jinja2Templates(directory=f"{Path('templates').resolve()}")

# functions to be used in templates as filters
templates.env.filters["format_timestamp"] = format_timestamp
templates.env.filters["sanitize_markdown"] = sanitize_markdown


def get_current_session(request: Request) -> dict[str, str]:
    if session_id := request.cookies.get("session_id"):
        with Session(engine) as session:
            user_session = session.get(UserSession, session_id)
            user_ = session.get(People, user_session.user_id)
        return {"session_id": session_id, "user_id": user_.user_id}
    else:
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            headers={"Location": router.url_path_for("user_login")},
        )


def set_session_cookie(response, session_id: str):
    with Session(engine) as session:
        user_expire_time = session.get(Config, "user_expire_time").value
    response.set_cookie(
        key="session_id", value=session_id, httponly=True, max_age=user_expire_time
    )  # max age accepts seconds
    return response


@router.get("/me")
async def get_user_me(
    request: Request,
    current_user: Dict[str, str] = Depends(get_current_session),
    session: Session = Depends(get_session),
):
    usr = session.get(People, current_user["user_id"])
    return templates.TemplateResponse("user_me.html", {"request": request, "usr": usr})


@router.get("/login", response_class=HTMLResponse, name="user_login")
async def login(request: Request):
    return templates.TemplateResponse("user_login.html", {"request": request})


@router.post("/login", response_class=HTMLResponse, name="user_login")
async def post_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    # Check if the user exists
    user_ = session.exec(select(People).where(People.email == email)).first()
    if not user_:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    # Check if the password hash is correct
    if user_.pass_hash != get_password_hash(user_.user_id, password, user_.salt):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)

    # Create a new session.
    new_session = UserSession(
        user_id=user_.user_id,
        login_time=datetime.now().timestamp(),
        expire_time=(datetime.now() + timedelta(days=60)).timestamp(),
        ip_address=request.client.host,
        user_agent=request.headers["User-Agent"],
    )
    session.add(new_session)
    session.commit()
    session.refresh(new_session)

    # Update user's last login time
    statement = select(People).where(People.user_id == user_.user_id)
    results = session.exec(statement)
    usr = results.one()
    usr.last_login = datetime.now().timestamp()
    session.add(usr)
    session.commit()

    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response = set_session_cookie(response, new_session.session_id)
    return response


@router.get("/logout", response_class=HTMLResponse, name="user_logout")
async def logout(request: Request, session: Session = Depends(get_session)):
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    session_id = request.cookies.get("session_id")

    # Delete the session from the database
    statement = select(UserSession).where(UserSession.session_id == session_id)
    results = session.exec(statement)
    user_session = results.one()
    session.delete(user_session)
    session.commit()

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


@router.post("/create", response_class=HTMLResponse, name="user_create")
async def post_user_create(
    display_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    cell_phone: str = Form(None),
    session: Session = Depends(get_session),
):
    salt = str(uuid4())
    new_user = People(
        display_name=display_name.title(),
        email=email.lower(),
        salt=salt,
        cell_phone=cell_phone,
        role="user",
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    new_user.pass_hash = get_password_hash(new_user.user_id, password, new_user.salt)
    session.add(new_user)
    session.commit()

    return f"Success!, new user id: {new_user.user_id}"
