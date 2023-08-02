"""Contains utilities used to setup effective tests."""

import random
import string
from typing import Dict

from fastapi.testclient import TestClient
from pydantic import EmailStr

from app.core.settings import settings


def random_lower_string(length: int = 32) -> str:
    """Generate a random lowercased string.

    Args:
        length: _description_. Defaults to 32.

    Returns:
        A random lowercased string with defined length.
    """
    return "".join(random.choices(string.ascii_lowercase, k=length))


def random_email() -> EmailStr:
    """Generate a random email address.

    Returns:
        A random email address.
    """
    return EmailStr(f"{random_lower_string(10)}@{random_lower_string(6)}.com")


def get_admin_token_headers(client: TestClient) -> Dict[str, str]:
    """Use the API route to get an admin access token header.

    The returned header is the one from the FIRST_ADMINUSER created on first start.

    Args:
        client: TestClient provided via fixture.

    Returns:
        Admin access token header of an admin user.
    """
    login_data = {
        "username": settings.FIRST_ADMINUSER,
        "password": settings.FIRST_ADMINUSER_PASSWORD,
    }
    r = client.post(f"{settings.API_V1_STR}/login/access-token", data=login_data)
    tokens = r.json()
    a_token = tokens["access_token"]
    headers = {"Authorization": f"Bearer {a_token}"}
    return headers
