from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, Session, SQLModel, create_engine

DATABASE_FILE = Path("events.db")
engine = create_engine(f"sqlite:///{DATABASE_FILE.resolve()}")


class Events(SQLModel, table=True):
    secret_code: str = Field(primary_key=True, index=True)
    active: bool = Field(default=True)
    user_id: str = Field(index=True, foreign_key="people.user_id")
    name: str
    virtual: bool = Field(default=False)
    url: Optional[str]
    location: Optional[str]
    details: Optional[str]
    created: int
    start_datetime: int = Field(index=True)
    end_datetime: Optional[int]


class People(SQLModel, table=True):
    user_id: str = Field(primary_key=True, index=True, default=str(uuid4()))
    active: bool = Field(default=False, index=True)
    display_name: Optional[str]
    email: Optional[str] = Field(index=True)
    salt: Optional[str]
    pass_hash: Optional[str]
    cell_phone: Optional[str]
    last_login: int = Field(default=datetime.now().timestamp())
    created: int = Field(default=datetime.now().timestamp())
    updated: int = Field(default=datetime.now().timestamp())
    role: str = Field(default="user", index=True)


class RSVP(SQLModel, table=True):
    rsvp_id: str = Field(primary_key=True, default=str(uuid4()))
    created: int
    user_id: str = Field(foreign_key="people.user_id", index=True)
    event_id: str = Field(foreign_key="events.secret_code", index=True)
    rsvp_status: str


class Config(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


class Tokens(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="people.user_id")
    token: str
    token_type: str  # Magic Link, Password Reset, API Key, etc.
    created_at: int = Field(default=datetime.now())
    key_expire_time: int = Field(
        default=int((datetime.now() + timedelta(days=60)).timestamp()),
        index=True,
    )
    updated_at: int = Field(default=datetime.now())


class UserSession(SQLModel, table=True):
    """
    Represents a user session in the application.

    Attributes:
        session_id (str): The unique identifier for the session. This is the primary key.
        user_id (str): The ID of the user who owns the session. This is a foreign key referencing the user's ID in the People table.
        login_time (int): The timestamp when the user logged in.
        expire_time (int): The timestamp when the session expires.
        ip_address (Optional[str]): The IP address from which the user logged in. This can be useful for security and auditing purposes.
        user_agent (Optional[str]): Information about the user's browser and operating system. This can also be useful for security and auditing purposes.
    """

    session_id: str = Field(primary_key=True, default=str(uuid4()))
    user_id: str = Field(foreign_key="people.user_id")
    login_time: int
    expire_time: int
    ip_address: Optional[str]
    user_agent: Optional[str]


def create_tables(engine_=engine):
    SQLModel.metadata.create_all(engine_)

    # Insert default config values
    with Session(engine_) as session:
        session.add(
            Config(key="user_expire_time", value=str(60 * 60 * 24 * 30))
        )  # 30 days
        session.add(
            Config(key="event_expire_time", value=str(60 * 60 * 24 * 90))
        )  # 90 days
        session.add(Config(key="secret_code_length", value=str(12)))  # 12 characters
        session.commit()


if __name__ == "__main__":
    # Create the tables
    create_tables()
