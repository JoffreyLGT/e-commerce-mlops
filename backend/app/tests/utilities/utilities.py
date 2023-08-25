"""Contains utilities used to setup effective tests."""

import random
import string

from fastapi.testclient import TestClient
from pydantic import EmailStr

from app.core.settings import settings
from datascience.src.data import CATEGORIES_DIC


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


def get_admin_token_headers(client: TestClient) -> dict[str, str]:
    """Use the API route to get an admin access token header.

    The returned header is the one from the FIRST_ADMINUSER created on first start.

    Args:
        client: TestClient provided via fixture.

    Returns:
        Admin access token header of an admin user.
    """
    login_data = {
        "username": str(settings.FIRST_ADMINUSER),
        "password": str(settings.FIRST_ADMINUSER_PASSWORD),
    }
    response = client.post(f"{settings.API_V1_STR}/login/access-token", data=login_data)
    tokens = response.json()
    a_token = tokens["access_token"]
    return {"Authorization": f"Bearer {a_token}"}


def random_category_id() -> int:
    """Return a random category id (prdtypecode).

    Returns:
        Random category id.
    """
    all_ids = list(CATEGORIES_DIC.keys())
    return random.choice(all_ids)
