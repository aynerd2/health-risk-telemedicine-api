"""
SQLModel engine and session dependency (replaces Django's ORM session handling).
"""
from collections.abc import Generator
from urllib.parse import parse_qs, urlsplit, urlunsplit

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings


def resolve_engine_url_and_connect_args(database_url: str) -> tuple[str, dict]:
    """Shared with migrations/env.py so Alembic's engine gets the same
    sqlite/mysql connect_args handling as the app's own engine."""
    if database_url.startswith("sqlite"):
        return database_url, {"check_same_thread": False}

    if database_url.startswith("mysql"):
        parsed = urlsplit(database_url)
        query = parse_qs(parsed.query)
        # PyMySQL's DBAPI has no `ssl-mode` connect() kwarg -- SQLAlchemy
        # passes URL query params straight through to the driver, so a
        # connection string with `?ssl-mode=REQUIRED` (what Aiven/PlanetScale/
        # etc. give you to copy-paste) fails with
        # "unexpected keyword argument 'ssl-mode'" instead of connecting.
        # Strip it and configure TLS via connect_args instead.
        if "ssl-mode" in query or "ssl_mode" in query:
            # verify_mode=False + check_hostname=False encrypts the
            # connection without validating the server's certificate --
            # matching ssl-mode=REQUIRED (not VERIFY_CA/VERIFY_IDENTITY, which
            # would need the provider's CA cert supplied via ssl_ca). Managed
            # MySQL providers commonly sign with a private per-service CA
            # that isn't in the system trust store, so full verification
            # isn't a drop-in replacement here.
            return urlunsplit(parsed._replace(query="")), {"ssl": {"ca": None, "check_hostname": False, "verify_mode": False}}

    return database_url, {}


_engine_url, connect_args = resolve_engine_url_and_connect_args(settings.DATABASE_URL)
engine = create_engine(_engine_url, echo=False, pool_pre_ping=True, connect_args=connect_args)


def init_db() -> None:
    """Create tables. In production, use Alembic migrations instead."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
