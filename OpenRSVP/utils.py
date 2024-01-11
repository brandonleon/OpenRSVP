from datetime import datetime
from uuid import uuid4
from icecream import ic


def code_format(st: str = None, ln: int = 12) -> str:
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
    pad = str(uuid4())
    pad = pad.replace("-", "")
    pad = pad[:ln]
    return f"{st}_{pad}"


# Function to take a unix timestamp and return a string in the default format YYYY-MM-DD HH:MM.
def format_timestamp(ts: int, frmt: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.fromtimestamp(ts).strftime(frmt)
