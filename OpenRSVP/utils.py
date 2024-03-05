"""
This module contains utility functions that are used throughout the application.
"""

from datetime import datetime
from typing import Union
from uuid import uuid4

import starlette
from bleach import clean
from markdown import markdown

from OpenRSVP.database import fetch_config, fetch_user, insert_user, update_user
from OpenRSVP.models import People, Config, engine
from sqlmodel import Session, select


def format_code_to_alphanumeric(st: str = None, ln: int = 12) -> str:
    """
    Ensures that a given string is at least 12 characters long.

    This function takes a string as an argument and checks if its length is less than 12 characters.
    If it is, the function generates a UUID, removes the hyphens from the UUID,
    appends the UUID to the string, and then truncates the string to be 12 characters long.
    The function also replaces spaces with underscores and removes all special characters,
    leaving only alphanumeric characters and numbers.

    Args:
        st (str): The input string.
        ln (int, optional): The minimum length of the string. Defaults to 12.

    Returns:
        str: The modified string that is guaranteed to be at least 12 characters long,
        containing only alphanumeric characters, numbers, and underscores.
    """
    if st is None:
        return str(uuid4())

    st = "".join([i for i in st if i.isalnum()])

    if len(st) < ln:
        padding = f"_{str(uuid4()).replace('-', '')}"
        st += padding[: ln - len(st)]
    return st


def pad_string(st: str, ln: int = 1) -> str:
    """
    Pads a string with a unique identifier.

    This function takes a string and a length as input. It generates a UUID, removes the hyphens,
    and truncates it to the specified length. This truncated UUID is then appended to the input string
    with an underscore separator. If the ln is 0, the function returns the input string as is.

    Args:
        st (str): The input string.
        ln (int, optional): The length of the padding. Defaults to 1.

    Returns:
        str: The input string padded with a unique identifier.
    """
    if ln == 0:
        return st
    pad = str(uuid4())
    pad = pad.replace("-", "")
    pad = pad[:ln]
    # Check if the original string has a _.
    return f"{st}{pad}" if "_" in st else f"{st}_{pad}"


# Function to take a unix timestamp and return a string in the default format YYYY-MM-DD HH:MM.
def format_timestamp(ts: Union[int, str], frmt: str = "%Y-%m-%d %H:%M") -> str:
    """
    Converts a Unix timestamp to a formatted string.

    This function takes a Unix timestamp and a format string as input. It converts the Unix timestamp
    to a datetime object, and then formats this datetime object into a string using the provided format string.

    Args:
        ts (int): The input Unix timestamp.
        frmt (str, optional): The format string used to format the datetime object. Defaults to "%Y-%m-%d %H:%M".

    Returns:
        str: The formatted datetime string.
    """
    if type(ts) is str and ts.isdigit():
        ts = int(ts)
    return datetime.fromtimestamp(ts).strftime(frmt)


def sanitize_markdown(md: str) -> str:
    """
    Converts Markdown text to HTML and sanitizes it.

    This function takes a markdown string as input, converts it to HTML using the markdown library,
    and then sanitizes the HTML using the bleach library. The sanitization process involves removing
    any HTML tags that are not in the specified list of allowed tags.

    Args:
        md (str): The input markdown string.

    Returns:
        str: The sanitized HTML string."""
    html = markdown(md)
    return clean(
        html,
        tags=[
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "br",
            "strong",
            "b",
            "em",
            "i",
            "blockquote",
            "ol",
            "li",
            "ul",
            "code",
            "hr",
            "a",
            "img",
            "pre",
            "del",
            "s",
            "sup",
            "sub",
            "table",
            "thead",
            "tbody",
            "tr",
            "th",
            "td",
        ],
    )


def get_user_id_from_cookie(request) -> Union[str, None]:
    """
    Fetches the user's ID from the request's cookies, if present.

    Args:
        request (Request): The input request object.

    Returns:
        Union[str, None]: The user's ID if present, otherwise None.
    """
    return user_id if (user_id := request.cookies.get("user_id")) else None


def get_or_set_user_id_cookie(request, response) -> starlette:
    """
    Fetches the user's ID from the request's cookies,
    if the cookie is not present, generates a new UUID and sets it in the cookies.
    If the user's ID is present in the cookies, updates the max_age of the cookie.

    Args:
        request (Request): The input request object.
        response (Response): The input response object.

    Returns:
        str: The user's ID.
    """
    with Session(engine) as session:
        user_expire_time = session.get(Config, "user_expire_time").value

    if not (user_id := get_user_id_from_cookie(request)):
        user_id = str(uuid4())

    # Set or update the cookie with the user_id
    response.set_cookie(
        key="user_id",
        value=user_id,
        httponly=True,
        max_age=user_expire_time,  # max age accepts seconds
        # expires=datetime.now() + timedelta(seconds=user_expire_time), # expires accepts a datetime object
    )

    with Session(engine) as session:
        user = session.get(People, user_id)
        user_id = user.user_id if user else user_id

    if not fetch_user(user_id):
        insert_user(user_id)
    else:
        update_user(user_id, "last_login", str(datetime.now().timestamp()))

    return response
