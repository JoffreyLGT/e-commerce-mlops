"""Test the prediction_feedback routes."""

import json
import random
from typing import Dict

from fastapi import status
from fastapi.testclient import TestClient

from app.core.settings import settings
from app.schemas import PredictionFeedback, PredictionFeedbackCreate
from app.tests.utilities.utilities import random_category_id


def test_get_feedbacks_admin(
    client: TestClient,
    admin_token_headers: Dict[str, str],
    prediction_feedback: PredictionFeedback,
) -> None:
    """Test the route to get feedback as admin.

    Information should be returned."""
    request = client.get(
        f"{settings.API_V1_STR}/feedback/", headers=admin_token_headers
    )
    assert request.status_code == status.HTTP_200_OK
    response = request.json()
    response_feedback = list(
        filter(lambda x: x["id"] == prediction_feedback.id, response)
    )
    assert len(response_feedback) == 1
    value = response_feedback[0]
    expected = prediction_feedback.to_json()
    assert value == expected


def test_get_feedbacks_user(
    client: TestClient, normal_user_token_headers: Dict[str, str]
) -> None:
    """Test the route to get feedback as normal user.

    Information should not be returned."""
    request = client.get(
        f"{settings.API_V1_STR}/feedback/", headers=normal_user_token_headers
    )
    assert (
        status.HTTP_400_BAD_REQUEST
        <= request.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def test_create_existing_feedback(
    client: TestClient,
    normal_user_token_headers: Dict[str, str],
    prediction_feedback: PredictionFeedback,
) -> None:
    """Test if we can add an already existing feedback.

    Should not be the case.
    """
    request = client.post(
        f"{settings.API_V1_STR}/feedback/",
        headers=normal_user_token_headers,
        json=prediction_feedback.to_json(),
    )
    assert (
        status.HTTP_400_BAD_REQUEST
        <= request.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def test_create_new_feedback(
    client: TestClient, normal_user_token_headers: Dict[str, str]
) -> None:
    """Test if we can create a new feedback."""
    expected = PredictionFeedbackCreate(
        product_id=random.randint(0, 999999),
        real_category_id=random_category_id(),
        pred_category_id=random_category_id(),
        model_version="1.0",
    )
    request = client.post(
        f"{settings.API_V1_STR}/feedback/",
        headers=normal_user_token_headers,
        json=expected.to_json(),
    )
    assert status.HTTP_200_OK <= request.status_code < status.HTTP_300_MULTIPLE_CHOICES
    result = request.json()
    # Remove id field since we can't predict it
    del result["id"]
    assert result == expected.to_json()
