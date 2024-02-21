from datetime import datetime
from uuid import uuid4

from bleach import clean
from markdown import markdown


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


def pad_string(st: str, ln: int = 6) -> str:
    """
    Pads a string with a unique identifier.

    This function takes a string and a length as input. It generates a UUID, removes the hyphens,
    and truncates it to the specified length. This truncated UUID is then appended to the input string
    with an underscore separator.

    Args:
        st (str): The input string.
        ln (int, optional): The length of the padding. Defaults to 6.

    Returns:
        str: The input string padded with a unique identifier.
    """
    pad = str(uuid4())
    pad = pad.replace("-", "")
    pad = pad[:ln]
    return f"{st}_{pad}"


# Function to take a unix timestamp and return a string in the default format YYYY-MM-DD HH:MM.
def format_timestamp(ts: int, frmt: str = "%Y-%m-%d %H:%M") -> str:
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
