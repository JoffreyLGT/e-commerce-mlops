"""Test the users routes."""


from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import crud
from app.core.settings import settings
from app.schemas.user import UserCreate
from app.tests.utilities.utilities import random_email, random_lower_string


def test_get_users_admin_me(
    client: TestClient, admin_token_headers: dict[str, str]
) -> None:
    """Test the route to retrieve admin user's own information."""
    request = client.get(f"{settings.API_V1_STR}/users/me", headers=admin_token_headers)
    assert status.HTTP_200_OK == request.status_code
    current_user = request.json()
    assert current_user
    assert current_user["is_active"] is True
    assert current_user["is_admin"]
    assert current_user["email"] == settings.FIRST_ADMINUSER


def test_get_users_normal_user_me(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    """Test the route to retrieve normal user's own information."""
    request = client.get(
        f"{settings.API_V1_STR}/users/me", headers=normal_user_token_headers
    )
    assert status.HTTP_200_OK == request.status_code

    current_user = request.json()
    assert current_user
    assert current_user["is_active"] is True
    assert current_user["is_admin"] is False
    assert current_user["email"] == settings.TEST_USER


def test_create_user_new_email(
    client: TestClient, admin_token_headers: dict[str, str], db: Session
) -> None:
    """Test the route to create a new user."""
    username = random_email()
    password = random_lower_string()
    data = {"email": username, "password": password}
    request = client.post(
        f"{settings.API_V1_STR}/users/",
        headers=admin_token_headers,
        json=data,
    )
    assert status.HTTP_200_OK <= request.status_code < status.HTTP_300_MULTIPLE_CHOICES
    created_user = request.json()
    user = crud.user.get_by_email(db, email=username)
    assert user
    assert user.email == created_user["email"]


def test_get_existing_user(
    client: TestClient, admin_token_headers: dict[str, str], db: Session
) -> None:
    """Test the route to get a specific user information."""
    username = random_email()
    password = random_lower_string()
    user_in = UserCreate(email=username, password=password)
    user = crud.user.create(db, obj_in=user_in)
    user_id = user.id
    request = client.get(
        f"{settings.API_V1_STR}/users/{user_id}",
        headers=admin_token_headers,
    )
    assert status.HTTP_200_OK == request.status_code
    api_user = request.json()
    existing_user = crud.user.get_by_email(db, email=username)
    assert existing_user
    assert existing_user.email == api_user["email"]


def test_create_user_existing_username(
    client: TestClient, admin_token_headers: dict[str, str], db: Session
) -> None:
    """Test the route to create a new user as admin with an existing id.

    An error must be returned since ids are unique.
    """
    username = random_email()
    # username = email
    password = random_lower_string()
    user_in = UserCreate(email=username, password=password)
    crud.user.create(db, obj_in=user_in)
    data = {"email": username, "password": password}
    request = client.post(
        f"{settings.API_V1_STR}/users/",
        headers=admin_token_headers,
        json=data,
    )
    assert (
        status.HTTP_400_BAD_REQUEST
        <= request.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    created_user = request.json()
    assert "_id" not in created_user


def test_create_user_by_normal_user(
    client: TestClient, normal_user_token_headers: dict[str, str]
) -> None:
    """Test the route to create a new user as normal user.

    Must return an error since only admin are allowed to do it.
    """
    username = random_email()
    password = random_lower_string()
    data = {"email": username, "password": password}
    request = client.post(
        f"{settings.API_V1_STR}/users/",
        headers=normal_user_token_headers,
        json=data,
    )
    assert (
        status.HTTP_400_BAD_REQUEST
        <= request.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def test_retrieve_users(
    client: TestClient, admin_token_headers: dict[str, str], db: Session
) -> None:
    """Test the route to retrieve users from DB."""
    username = random_email()
    password = random_lower_string()
    user_in = UserCreate(email=username, password=password)
    crud.user.create(db, obj_in=user_in)

    username2 = random_email()
    password2 = random_lower_string()
    user_in2 = UserCreate(email=username2, password=password2)
    crud.user.create(db, obj_in=user_in2)

    request = client.get(f"{settings.API_V1_STR}/users/", headers=admin_token_headers)
    assert status.HTTP_200_OK == request.status_code
    all_users = request.json()

    assert len(all_users) > 1
    for item in all_users:
        assert "email" in item
