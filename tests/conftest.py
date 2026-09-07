import os
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.core.bootstrap import create_admin_user
from app.core.database import engine as real_engine
from app.core.database import get_session
from app.main import app
from app.models.entities import Condition
from app.services import prediction_service

TEST_AGAINST_MYSQL = os.environ.get("TEST_AGAINST_MYSQL") == "1"


@pytest.fixture(name="session")
def session_fixture() -> Generator[Session, None, None]:
    if TEST_AGAINST_MYSQL:
        # Run against the real DATABASE_URL (see app/core/database.py) instead
        # of a disposable SQLite DB. Tables already exist via `alembic upgrade
        # head`, and most tests reuse the same fixed emails across many test
        # functions in the same file, so each test gets its own transaction
        # that's rolled back at teardown -- true per-test isolation against a
        # shared persistent database, and nothing is left behind afterward.
        connection = real_engine.connect()
        transaction = connection.begin()
        # join_transaction_mode="create_savepoint": route handlers call
        # session.commit() freely (e.g. register() commits the User, then
        # commits again for the Patient/Doctor row) -- without this, that
        # commit() would commit the *outer* transaction too, and the
        # rollback() below would only undo whatever happened after the last
        # commit, not the whole test. With it, commit() only releases a
        # SAVEPOINT; verified this empirically before trusting it here.
        session = Session(bind=connection, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            session.close()
            transaction.rollback()
            connection.close()
        return

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session) -> Generator[TestClient, None, None]:
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def stub_ml_models(client: TestClient):
    """
    The three .pkl files aren't part of this repo until real datasets are
    trained on (see BUILD_PLAN.md step 4) — stand in a dummy model that always
    predicts a fixed probability so the prediction endpoints are testable now.
    Depends on `client` so it overwrites what the app's startup lifespan
    (load_models(), which finds no .pkl files and sets None) just loaded.
    """
    dummy = MagicMock()
    dummy.predict_proba.return_value = [[0.2, 0.8]]
    prediction_service._MODELS.update({condition: dummy for condition in Condition})
    yield
    prediction_service._MODELS.clear()


@pytest.fixture
def admin_token(client: TestClient, session: Session) -> str:
    create_admin_user(session, email="admin@example.com", password="adminpass123", full_name="Admin")
    resp = client.post("/api/v1/auth/login", data={"username": "admin@example.com", "password": "adminpass123"})
    assert resp.status_code == 200
    return resp.json()["access_token"]
