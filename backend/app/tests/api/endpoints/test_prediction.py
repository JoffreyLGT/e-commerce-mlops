from io import BytesIO
from typing import Dict
from fastapi import status
from app.core.settings import settings
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


# client = TestClient(app)


def test_predict_category_with_valid_data(
    client: TestClient, normal_user_token_headers: Dict[str, str]
):
    """Test the prediction with valid and wrong data, depending the case"""

    # sending data request
    designation = "designation"
    description = "product description."
    image = Image.new("RGB", (100, 100))  # test image uploaded in color
    image_file = (
        BytesIO()
    )  #  BytesIO is use to simulate image file. this is where we save the image test
    image.save(image_file, format="jpeg")  # we save the test image in jpeg format

    # test with positive limit value
    limit = 3
    # send POST request
    request = client.post(
        f"{settings.API_V1_STR}/predict/",
        params={"designation": designation, "description": description, "limit": limit},
        headers=normal_user_token_headers,
        files={"image": ("product.jpg", image_file, "image/jpeg")},
    )
    assert request.status_code == status.HTTP_200_OK

    # Test with negative limit value
    limit_negative = -1

    # sending POST request with negative limit value
    request_negative_limit = client.post(
        f"{settings.API_V1_STR}/predict/",
        params={
            "designation": designation,
            "description": description,
            "limit": limit_negative,
        },
        headers=normal_user_token_headers,
        files={"image": ("product.jpg", image_file, "image/jpeg")},
    )

    # verify if the status code is 400 (Bad Request)
    assert request_negative_limit.status_code == status.HTTP_400_BAD_REQUEST
    # verifing that error detail for wrong limit type is correct
    error_detail = request_negative_limit.json()["detail"]
    assert "Invalid value for limit. Must be a positive integer." in error_detail

    # Test with wrong image extension (PNG instead of JPEG or JPG)
    image_file_png = (
        BytesIO()
    )  # BytesIO is use to simulate image file. this is where we save the image test
    image.save(image_file_png, format="png")  # we save the test image in png format

    # sending POST request with wrong image extension
    request_invalid_extension = client.post(
        f"{settings.API_V1_STR}/predict/",
        params={"designation": designation, "description": description},
        headers=normal_user_token_headers,
        files={"image": ("product.png", image_file_png, "image/png")},
    )

    # verify if the status code is 400 (Bad Request)
    assert request_invalid_extension.status_code == status.HTTP_400_BAD_REQUEST
    # verifing that error detail for wrong image extension is correct
    error_detail_invalid_extension = request_invalid_extension.json()["detail"]
    assert (
        "Invalid image extension. Image must be in JPEG or JPG format."
        in error_detail_invalid_extension
    )


def test_predict_category_with_invalid_image_format(client: TestClient):
    """test of the prediction with wrong image format;
    this should not be the case and for that ,
    must raire an error"""

    # data for the test
    designation = "designation"
    description = "product description."
    image_data = Image.new("RGB", (100, 100))  # test image uploaded in color
    image_file = (
        BytesIO()
    )  #  BytesIO is use to simulate image file. this is where we save the image test
    image_data.save(image_file, format="png")  # we save the test image in png format

    # sending POST request with incorrect image format
    request = client.post(
        f"{settings.API_V1_STR}/predict/",
        data={"designation": designation, "description": description},
        files={"image": ("product.png", image_file, "image/png")},
    )

    # verify the response code for invalid request
    assert request.status_code == status.HTTP_401_UNAUTHORIZED


def test_predict_category_without_data(client: TestClient):
    """Test  the prediction with empty data
    Must raise an error"""

    # Sending POST request without any data
    request_without_data = client.post(f"{settings.API_V1_STR}/predict/")

    # verify the response code for invalid request
    assert request_without_data.status_code == status.HTTP_401_UNAUTHORIZED
