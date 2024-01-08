import sqlite3


import sqlite3


def init_db() -> tuple[bool, str | None]:
    """
    Initialize the database
    :return: Tuple with True if successful, False if not, and an error message if any
    """
    try:
        conn = sqlite3.connect("events.db")
        c = conn.cursor()
        c.execute(
            "CREATE TABLE IF NOT EXISTS events (event_id text, name text, date text, start_time text, end_time text secret_code text)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS people (user_id text, name text, email text)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS rsvp (rsvp_id text, user_id text, event_id text, rsvp_status text)"
        )
        conn.commit()
        conn.close()
        return True, None  # Successful initialization, no error message
    except Exception as e:
        error_message = str(e)
        return False, error_message
