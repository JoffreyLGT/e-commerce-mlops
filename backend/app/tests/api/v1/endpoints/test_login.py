"""Contains the tests on login routes."""


from fastapi import status
from fastapi.testclient import TestClient

from app.core.settings import settings


def test_get_access_token(client: TestClient) -> None:
    """Test the recuperation of an access token."""
    login_data = {
        "username": settings.FIRST_ADMINUSER,
        "password": settings.FIRST_ADMINUSER_PASSWORD,
    }
    response = client.post(f"{settings.API_V1_STR}/login/access-token", data=login_data)
    assert response.status_code == status.HTTP_200_OK
    tokens = response.json()
    assert "access_token" in tokens
    assert tokens["access_token"]


def test_use_access_token(
    client: TestClient, admin_token_headers: dict[str, str]
) -> None:
    """Test if the authentication is working by validating the access token."""
    response = client.post(
        f"{settings.API_V1_STR}/login/test-token",
        headers=admin_token_headers,
    )
    assert response.status_code == status.HTTP_200_OK
    result = response.json()
    assert "email" in result
