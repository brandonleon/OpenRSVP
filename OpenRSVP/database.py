import sqlite3
from pathlib import Path
from time import time

CURRENT_DIR = Path(__file__).parent
DATABASE_FILE = Path("events.db")
SCHEMA_FILE = CURRENT_DIR / ".." / "sql" / "schema.sql"


def init_db() -> tuple[bool, str | None]:  # sourcery skip: extract-method
    """
    Initializes the database by creating the necessary tables if they do not exist.

    This function attempts to connect to the SQLite database 'events.db' and creates three tables:
    'events', 'people', and 'rsvp' if they do not already exist.

    The 'events' table has the following columns: event_id, name, date, start_time, end_time, and secret_code.
    The 'people' table has the following columns: user_id, name, and email.
    The 'rsvp' table has the following columns: rsvp_id, user_id, event_id, and rsvp_status.

    If the database initialization is successful, the function returns a tuple (True, None).
    If an exception occurs during the initialization, the function catches the exception, converts it to a string,
    and returns a tuple (False, error_message) where error_message is the string representation of the exception.

    Returns:
        tuple[bool, str | None]: A tuple where the first element is a boolean indicating the success of the operation,
        and the second element is either None (if successful) or a string containing an error message (if unsuccessful).
    """
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()

            with open(SCHEMA_FILE) as f:
                c.executescript(f.read())
            conn.commit()

            # insert default config values
            insert_config("user_expire_time", str(60 * 60 * 24 * 90))  # 90 days
            insert_config(
                "event_expire_time", str(60 * 60 * 24 * 180)
            )  # 180 days or approximately 6 months.
        return True, None  # Successful initialization, no error message
    except Exception as e:
        error_message = str(e)
        return False, error_message


def insert_event(
    code, name, user_id, details, start_datetime, end_datetime
) -> tuple[bool, str | None]:
    """
    Inserts a new event into the 'events' table in the SQLite database 'events.db'.

    This function takes six arguments: name, start_date, start_time, end_date, end_time, and secret_code.
    It connects to the SQLite database 'events.db', creates a cursor,
    and executes an SQL INSERT INTO command to insert a new row into the 'events' table with the provided values.
    After executing the command, it commits the changes and closes the connection to the database.

    Args:
        code (str): The secret code associated with the event.
        name (str): The name of the event.
        user_id (str): The user ID of the event creator.
        details (str): The details of the event.
        start_datetime (str): The start datetime of the event in ISO 8601 format.
        end_datetime (str): The end datetime of the event in ISO 8601 format.

    Returns:
        tuple[bool, str | None]: A tuple where the first element is a boolean indicating the success of the operation,
        and the second element is either None (if successful) or a string containing an error message (if unsuccessful).
    """
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()

            # If start_datetime is in the future, the event is active
            active = 1 if int(start_datetime) > int(time()) else 0

            c.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    active,
                    code,
                    name,
                    user_id,
                    details,
                    start_datetime,
                    end_datetime,
                ),
            )
            conn.commit()
        return True, None  # Successful insertion, no error message
    except Exception as e:
        error_message = str(e)
        return False, error_message  # Unsuccessful insertion, error message


def fetch_event(code: str) -> dict | bool:
    """
    Fetches an event from the database.

    Args:
        code (str): The event's secret code.

    Returns:
        dict | bool: A dictionary containing the event's details if the operation is successful,
        or False if an exception occurs.
    """

    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT secret_code, name, details, start_datetime, end_datetime FROM events WHERE secret_code = ?",
                (code,),
            )
            event = c.fetchone()
            return {
                "secret_code": event[0],
                "name": event[1],
                "details": event[2],
                "start_datetime": event[3],
                "end_datetime": event[4],
            }
    except Exception as e:
        print(e)
        return False


def insert_user(user_id: str) -> tuple[bool, str | None]:
    """
    Inserts a new user into the 'people' table in the SQLite database 'events.db'.

    This function takes one argument: user_id.
    It connects to the SQLite database 'events.db', creates a cursor,
    and executes an SQL INSERT INTO command to insert a new row into the 'people' table with the provided user_id.
    After executing the command, it commits the changes and closes the connection to the database.

    Args:
        user_id (str): The user's ID.

    Returns:
        tuple[bool, str | None]: A tuple where the first element is a boolean indicating the success of the operation,
        and the second element is either None (if successful) or a string containing an error message (if unsuccessful).
    """
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO people (user_id, last_login, created, updated, role) VALUES (?, ?, ?, ?, ?)",
                (user_id, int(time()), int(time()), int(time()), "user"),
            )
            conn.commit()
        return True, None  # Successful insertion, no error message
    except Exception as e:
        error_message = str(e)
        return False, error_message  # Unsuccessful insertion, error message


def fetch_user(user_id: str) -> dict | bool:
    """
    Fetches a user from the database.

    Args:
        user_id (str): The user's ID.

    Returns:
        dict | bool: A dictionary containing the user's details if the operation is successful,
        or False if an exception occurs.
    """

    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT user_id, display_name, email, cell_phone FROM people WHERE user_id = ?",
                (user_id,),
            )
            user = c.fetchone()
            return {
                "user_id": user[0],
                "name": user[1],
                "email": user[2],
                "cell_phone": user[3],
            }
    except Exception as e:
        print(e)
        return False


def update_user(user_id: str, column: str, value: str) -> tuple[bool, str | None]:
    """
    Updates a user's details in the 'people' table in the SQLite database 'events.db'.

    This function takes three arguments: user_id, column, and value.
    It connects to the SQLite database 'events.db', creates a cursor,
    and executes an SQL UPDATE command to update the specified column of the user with the provided user_id.
    After executing the command, it commits the changes and closes the connection to the database.

    Args:
        user_id (str): The user's ID.
        column (str): The column to be updated.
        value (str): The new value of the column.

    Returns:
        tuple[bool, str | None]: A tuple where the first element is a boolean indicating the success of the operation,
        and the second element is either None (if successful) or a string containing an error message (if unsuccessful).
    """
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute(
                f"UPDATE people SET {column} = ? WHERE user_id = ?",
                (value, user_id),
            )
            conn.commit()
        return True, None  # Successful update, no error message
    except Exception as e:
        error_message = str(e)
        return False, error_message  # Unsuccessful update, error message


def fetch_config(key: str) -> str | bool:
    """
    Fetches a config value from the database.

    Args:
        key (str): The config value's key.

    Returns:
        str | bool: A string containing the config value if the operation is successful,
        or False if an exception occurs.
    """

    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT value FROM config WHERE key = ?",
                (key,),
            )
            value = c.fetchone()
            return value[0]
    except Exception as e:
        print(e)
        return False


# insert default config values
def insert_config(key: str, value: str) -> tuple[bool, str | None]:
    """
    Inserts a new config value into the 'config' table in the SQLite database 'events.db'.

    This function takes two arguments: key and value.
    It connects to the SQLite database 'events.db', creates a cursor,
    and executes an SQL INSERT INTO command to insert a new row into the 'config' table with the provided values.
    After executing the command, it commits the changes and closes the connection to the database.

    Args:
        key (str): The key of the config value.
        value (str): The value of the config value.

    Returns:
        tuple[bool, str | None]: A tuple where the first element is a boolean indicating the success of the operation,
        and the second element is either None (if successful) or a string containing an error message (if unsuccessful).
    """
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO config VALUES (?, ?)", (key, value))
            conn.commit()
        return True, None  # Successful insertion, no error message
    except Exception as e:
        error_message = str(e)
        return False, error_message  # Unsuccessful insertion, error message
