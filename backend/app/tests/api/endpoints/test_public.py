"""Test public routes."""

from fastapi.testclient import TestClient

from app.core.settings import settings
from app.main import app

client = TestClient(app)


def test_is_online():
    """Test if the API is online."""
    response = client.get(f"{settings.API_V1_STR}/")
    assert response.status_code == 200
    assert response.json() == [f"API {settings.PROJECT_NAME} is online!"]
