"""Contains all the fixtures used in tests."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.settings import settings
from app.database.session import SessionLocal
from app.main import app
from app.schemas import PredictionFeedback
from app.tests.utilities.prediction_feedback import create_random_feedback
from app.tests.utilities.user import authentication_token_from_email
from app.tests.utilities.utilities import get_admin_token_headers


@pytest.fixture(name="db", scope="session")
def fixture_db() -> Generator[Session, None, None]:
    """Fixture to get DB session.

    Yields:
        Generator of a DB session.
    """
    with SessionLocal() as session:
        yield session


@pytest.fixture(name="client", scope="module")
def fixture_client() -> Generator[TestClient, None, None]:
    """Fixture to get a TestClient.

    Yields:
        Generator of a TestClient.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def admin_token_headers(client: TestClient) -> dict[str, str]:
    """Fixture to get an admin token header.

    The returned header is the one from the FIRST_ADMINUSER created on first start.

    Args:
        client: TestClient resolved via fixture.

    Returns:
        FIRST_ADMINUSER admin token header.
    """
    return get_admin_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> dict[str, str]:
    """Fixture to get a normal user token header.

    The returned header is the one from the TEST_USER.
    TEST_USER is created if it doesn't exist in DB.

    Args:
        client: TestClient resolved via fixture.
        db: DB Session resolved via fixture.

    Returns:
        TEST_USER normal user header.
    """
    return authentication_token_from_email(
        client=client, email=settings.TEST_USER, db=db
    )


@pytest.fixture(scope="module")
def prediction_feedback(db: Session) -> PredictionFeedback:
    """Insert a new prediction feedback in DB and returns it.

    Args:
        db: database session.

    Returns:
        Newly created prediction feedback.
    """
    return create_random_feedback(db=db)
