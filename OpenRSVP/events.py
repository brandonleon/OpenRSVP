import datetime
import sqlite3
import uuid


def create_event(name, date, start_time, end_time, secret_code=None) -> str:
    """
    Creates an event in the database and returns the secret code, if one is not provided it will generate one
    :param name: Name of the event
    :param date: Date of the event
    :param start_time: Start time of the event
    :param end_time: End time of the event
    :param secret_code: Secret code for the event
    :return: Secret code for the event
    """
    if secret_code is None:
        secret_code = str(uuid.uuid4())
    elif len(secret_code) < 12:  # TODO: Make this a config variable
        secret_code = secret_code + str(uuid.uuid4())
        secret_code = secret_code[:12]
    else:
        secret_code = secret_code
    conn = sqlite3.connect("events.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
        (secret_code, name, date, start_time, end_time),
    )
    conn.commit()
    conn.close()
    return secret_code
