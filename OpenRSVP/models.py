from pathlib import Path
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel, create_engine, Session

DATABASE_FILE = Path("events.db")
engine = create_engine(f"sqlite:///{DATABASE_FILE.resolve()}")


class Events(SQLModel, table=True):
    active: bool = Field(default=True)
    secret_code: str = Field(primary_key=True, index=True)
    user_id: str = Field(index=True)
    name: str
    virtual: bool = Field(default=False)
    url: Optional[str]
    location: Optional[str]
    details: Optional[str]
    created: int
    start_datetime: int = Field(index=True)
    end_datetime: Optional[int]


class People(SQLModel, table=True):
    user_id: str = Field(primary_key=True, index=True, default=uuid4())
    display_name: Optional[str]
    email: Optional[str] = Field(index=True)
    salt: Optional[str]
    pass_hash: Optional[str]
    cell_phone: Optional[str]
    last_login: int
    created: int
    updated: int
    role: str = Field(default="user", index=True)


class RSVP(SQLModel, table=True):
    rsvp_id: str = Field(primary_key=True, default=uuid4())
    created: int
    user_id: str
    event_id: str
    rsvp_status: str


class Config(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


class UserSession(SQLModel, table=True):
    session_id: int = Field(primary_key=True, default=uuid4())
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
