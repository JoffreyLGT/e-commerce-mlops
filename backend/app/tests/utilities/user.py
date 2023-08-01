"""Utilities regarding user management used in tests."""
from typing import Dict

from fastapi.testclient import TestClient
from pydantic import EmailStr
from sqlalchemy.orm import Session

from app import crud
from app.core.settings import settings
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from app.tests.utilities.utilities import random_email, random_lower_string


def get_user_authentication_headers(
    *, client: TestClient, email: str, password: str
) -> Dict[str, str]:
    """Get user authentication headers.

    Args:
        client: test client to interract with the app.
        email: user email.
        password: user password.

    Returns:
        User authorization headers.
    """
    data = {"username": email, "password": password}

    response = client.post(f"{settings.API_V1_STR}/login/access-token", data=data)
    response = response.json()
    auth_token = response["access_token"]
    headers = {"Authorization": f"Bearer {auth_token}"}
    return headers


def create_random_user(db: Session) -> User:
    """Create a user with random information in DB.

    Args:
        db: database session.

    Returns:
        New user information.
    """
    email = random_email()
    password = random_lower_string()
    user_in = UserCreate(email=email, password=password)
    user = crud.user.create(db=db, obj_in=user_in)
    return user


def authentication_token_from_email(
    *, client: TestClient, email: str, db: Session
) -> Dict[str, str]:
    """Return a valid token for the user with given email.

    If the user doesn't exist, it is created first.
    If the user exists, a new password is set so we can return a token.

    Args:
        client: test client to interract with the app.
        email: user email.
        db: database session.

    Returns:
        User authorization headers.
    """
    password = random_lower_string()
    user = crud.user.get_by_email(db, email=email)
    if not user:
        user_in_create = UserCreate(email=EmailStr(email), password=password)
        user = crud.user.create(db, obj_in=user_in_create)
    else:
        user_in_update = UserUpdate(password=password)
        user = crud.user.update(db, db_obj=user, obj_in=user_in_update)

    return get_user_authentication_headers(
        client=client, email=email, password=password
    )
