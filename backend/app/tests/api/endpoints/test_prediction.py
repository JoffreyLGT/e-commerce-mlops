from io import BytesIO
from typing import Dict
from fastapi import status
from app.core.settings import settings
from fastapi.testclient import TestClient
from PIL import Image


def test_predict_category_with_valid_data(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test the prediction with valid  data."""

    # request with color image
    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": 7,
    }
    image = Image.new("RGB", (450, 1470))  # test image uploaded in color
    image_file = (
        BytesIO()
    )  #  BytesIO is use to simulate image file. this is where we save the image test
    image.save(image_file, format="jpeg")  # we save the test image in jpeg format

    request = client.post(
        f"{settings.API_V1_STR}/predict/",
        params=params,
        headers=normal_user_token_headers,
        files={"image": ("product.jpg", image_file, "image/jpeg")},
    )
    assert request.status_code == status.HTTP_200_OK

    # request with black_and_white image
    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": 6,
    }
    image = Image.new(
        "RGB", (4785, 6370), color=0
    )  # test image uploaded in black_and_white, the "0" passed throught color is to tell us that canal color is black
    image_file = (
        BytesIO()
    )  #  BytesIO is use to simulate image file. this is where we save the image test
    image.save(image_file, format="jpeg")  # we save the test image in jpeg format

    request = client.post(
        f"{settings.API_V1_STR}/predict/",
        params=params,
        headers=normal_user_token_headers,
        files={"image": ("product.jpg", image_file, "image/jpeg")},
    )
    assert request.status_code == status.HTTP_200_OK


def test_predict_category_with_negative_limit(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test the prediction with negative limit value.

    this should not be the case!"""

    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": -1,
    }

    image = Image.new("RGB", (1784, 4586))
    image_file = (
        BytesIO()
    )  #  BytesIO is use to simulate image file. this is where we save the image test
    image.save(image_file, format="jpeg")  # we save the test image in jpeg format

    request_negative_limit = client.post(
        f"{settings.API_V1_STR}/predict/",
        params=params,
        headers=normal_user_token_headers,
        files={"image": ("product.jpg", image_file, "image/jpeg")},
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
    """Test the prediction with  wrong extension."""

    # sending data request
    params = {
        "designation": "Guillaume Le Maréchal Tome 1 - Le Chevalier D'aliénor",
        "description": "L'histoire oubliée du meilleur chevalier du monde... 1167, duché de Normandie..",
        "limit": 7,
    }
    image = Image.new("RGB", (3245, 5641))  # test image uploaded in color

    # Test with wrong image extension (PNG instead of JPEG or JPG)
    image_file_png = (
        BytesIO()
    )  # BytesIO is use to simulate image file. this is where we save the image test
    image.save(image_file_png, format="png")  # we save the test image in png format

    request_invalid_extension = client.post(
        f"{settings.API_V1_STR}/predict/",
        params=params,
        headers=normal_user_token_headers,
        files={"image": ("product.png", image_file_png, "image/png")},
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


def test_predict_category_with_invalid_image_format(
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
    image_data = Image.new("RGB", (100, 100))  # test image uploaded in color
    image_file = (
        BytesIO()
    )  #  BytesIO is use to simulate image file. this is where we save the image test
    image_data.save(image_file, format="png")  # we save the test image in png format

    # sending POST request with incorrect image format
    request_invalid_format = client.post(
        f"{settings.API_V1_STR}/predict/",
        headers=normal_user_token_headers,
        params=params,
        files={"image": ("product.png", image_file, "image/png")},
    )

    assert (
        status.HTTP_400_BAD_REQUEST
        <= request_invalid_format.status_code
        < status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    error_detail_invalid_format = request_invalid_format.json()["detail"]
    assert (
        "Invalid image format. Image must be in JPEG or JPG format."
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
