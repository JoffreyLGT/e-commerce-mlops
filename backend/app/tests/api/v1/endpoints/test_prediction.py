"""Test the prediction routes."""

from io import BytesIO
from typing import Dict

from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from app.core.settings import settings


def test_predict_category_valid_color(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test the prediction with valid data and a RGB picture."""

    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": 1,
    }
    image = Image.new("RGB", (450, 1470))
    image_file = BytesIO()  #  BytesIO is use to simulate image file
    image.save(image_file, format="jpeg")

    request = client.post(
        f"{settings.API_V1_STR}/predict/",
        params=params,
        headers=normal_user_token_headers,
        files={"image": ("product.jpg", image_file, "image/jpeg")},
    )
    assert request.status_code == status.HTTP_200_OK

    response = request.json()
    assert len(response) == params["limit"]

    assert (
        response[0]["prdtypecode"] == 1180
        and response[0]["probabilities"] == 45.0
        and response[0]["label"] == "Univers fantastiques"
    )


def test_predict_category_invalid_gray(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test the prediction with valid text but black and white image.."""

    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": 7,
    }
    image = Image.new(
        "L", (4785, 6370)
    )  # test image uploaded in black_and_white, the "0" passed throught color is to tell us that canal color is black
    image_file = BytesIO()  #  BytesIO is use to simulate image file
    image.save(image_file, format="jpeg")

    request = client.post(
        f"{settings.API_V1_STR}/predict/",
        params=params,
        headers=normal_user_token_headers,
        files={"image": ("product.jpg", image_file, "image/jpeg")},
    )
    assert (
        status.HTTP_400_BAD_REQUEST
        <= request.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )


def test_predict_category_with_negative_limit(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test the prediction with negative limit value.

    Should return an error."""

    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": -1,
    }

    request_negative_limit = client.post(
        f"{settings.API_V1_STR}/predict/",
        params=params,
        headers=normal_user_token_headers,
    )

    assert (
        status.HTTP_400_BAD_REQUEST
        <= request_negative_limit.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    error_detail = request_negative_limit.json()["detail"]
    assert "Invalid value for limit. Must be a positive integer." in error_detail


def test_predict_category_with_wrong_extension(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test the prediction with wrong extension."""

    # sending data request
    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": 7,
    }
    image = Image.new("RGB", (3245, 5641))

    image_file = BytesIO()  #  BytesIO is use to simulate image file
    image.save(image_file, format="png")

    request_invalid_extension = client.post(
        f"{settings.API_V1_STR}/predict/",
        params=params,
        headers=normal_user_token_headers,
        files={"image": ("product.png", image_file, "image/png")},
    )

    assert (
        status.HTTP_400_BAD_REQUEST
        <= request_invalid_extension.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    error_detail_invalid_extension = request_invalid_extension.json()["detail"]
    assert (
        "Invalid image extension. Image must be in JPEG or JPG format."
        in error_detail_invalid_extension
    )


def test_predict_category_invalid_image_format(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test with wrong image format."""

    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": 7,
    }

    image_file = BytesIO()
    image_file.write(b"Hello world")

    # sending POST request with incorrect image format
    request_invalid_format = client.post(
        f"{settings.API_V1_STR}/predict/",
        headers=normal_user_token_headers,
        params=params,
        files={"image": ("product.jpg", image_file, "image/jpg")},
    )

    assert (
        status.HTTP_400_BAD_REQUEST
        <= request_invalid_format.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    error_detail_invalid_format = request_invalid_format.json()["detail"]
    assert (
        "Invalid image format. Image must be in JPEG or JPG format and colored."
        in error_detail_invalid_format
    )


def test_predict_category_image_four_dimensions(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """test of the prediction with wrong image format,

    this should not be the case and for that ,
    must raire an error.
    """

    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": 7,
    }
    image_data = Image.new("RGBA", (100, 100))  # test image uploaded in color
    image_file = (
        BytesIO()
    )  #  BytesIO is use to simulate image file. this is where we save the image test
    image_data.save(image_file, format="png")  # we save the test image in png format

    # sending POST request with incorrect image format
    request_invalid_format = client.post(
        f"{settings.API_V1_STR}/predict/",
        headers=normal_user_token_headers,
        params=params,
        files={"image": ("product.jpg", image_file, "image/jpg")},
    )

    assert (
        status.HTTP_400_BAD_REQUEST
        <= request_invalid_format.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    error_detail_invalid_format = request_invalid_format.json()["detail"]
    assert (
        "Invalid image format. Image must be in JPEG or JPG format and colored."
        in error_detail_invalid_format
    )


def test_predict_category_without_data(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test  the prediction with empty data

    Must raise an error"""

    request_without_data = client.post(
        f"{settings.API_V1_STR}/predict/",
        headers=normal_user_token_headers,
    )

    assert (
        status.HTTP_400_BAD_REQUEST
        <= request_without_data.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )

    error_detail_without_data = request_without_data.json()["detail"]
    assert (
        "You must provide either designation/description or an image to get a result."
        in error_detail_without_data
    )
