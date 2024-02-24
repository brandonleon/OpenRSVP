from pathlib import Path
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel, create_engine

DATABASE_FILE = Path("events.db")
engine = create_engine(f"sqlite:///{DATABASE_FILE.resolve()}", echo=True)


class Event(SQLModel, table=True):
    active: int = Field(default=1)
    secret_code: str = Field(primary_key=True, index=True)
    name: str
    user_id: str = Field(index=True)
    details: Optional[str]
    start_datetime: int = Field(index=True)
    end_datetime: Optional[int]


class Person(SQLModel, table=True):
    user_id: str = Field(primary_key=True, index=True, default=uuid4())
    display_name: Optional[str]
    email: Optional[str] = Field(index=True)
    salt: Optional[str]
    password: Optional[str]
    cell_phone: Optional[str]
    last_login: int
    created: int
    updated: int
    role: str = Field(default="user", index=True)


class RSVP(SQLModel, table=True):
    rsvp_id: str = Field(primary_key=True, default=uuid4())
    user_id: str
    event_id: str
    rsvp_status: str


class Config(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


def create_tables(engine=engine):
    SQLModel.metadata.create_all(engine)


if __name__ == "__main__":
    create_tables()
