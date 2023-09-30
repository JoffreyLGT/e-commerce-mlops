"""Routes to predict product categories."""

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app import models
from app.api import dependencies
from app.core import image_data_utils
from app.core.settings import settings
from app.schemas import PredictionResult

router = APIRouter()


@router.post("/", response_model=list[PredictionResult])
async def predict_category(
    *,
    designation: str | None = "",
    description: str | None = "",
    image: UploadFile | None = None,
    limit: int | None = None,
    # Authentication and access management, do not delete!
    _current_user: models.User = Depends(  # pyright: ignore
        dependencies.get_current_active_user
    ),
) -> Any:
    """Predict the category of the product.

    Raises:
    - HTTPException: 400 if image extension is invalid.
    - HTTPException: 400 if image format is invalid.
    - HTTPException: 400 if limit is not a positive int.
    - HTTPException: 400 if all data fields are empty.
    - HTTPException: 500 if an error occures when requesting prediction.

    Returns:
        A list of object with 2 properties: category_id and probabilities.
        It is sorted by probabilities descending.
    """
    expected_shape_len = 3
    image_path = ""
    if image is not None:
        extension = (
            False
            if image.filename is None
            else image.filename.split(".")[-1] in ("jpg", "jpeg")
        )
        if extension is False:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="Invalid image extension. Image must be in JPEG or JPG format.",
            )
        try:
            # Open the image with PIL
            image_data = np.asarray(Image.open(BytesIO(await image.read())))
        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Invalid image format. Image must be "
                    "in JPEG or JPG format and colored."
                ),
            ) from exc

        # Case where the image is in Black and white or with another format
        if (
            len(image_data.shape) != expected_shape_len
            or image_data.shape[2] != expected_shape_len
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=("Invalid image: should be a coloured JPEG or JPG."),
            )

        image_path = str(image_data_utils.save_temporary_image(image_data))

    if limit is not None and limit <= 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Invalid value for limit. Must be a positive integer.",
        )

    # Check if we have data. We need either text data or an image to do a prediction
    if designation == "" and description == "" and image is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="You must provide either designation/description or an image to get a result.",  # noqa: E501
        )

    rqt_body = {
        "dataframe_records": [
            {
                "product_id": "1",
                "designation": designation,
                "description": description,
                "image_path": image_path,
            }
        ]
    }

    response = requests.post(
        (
            f"http://{settings.DATASCIENCE_SERVER}:{settings.DATASCIENCE_MODEL_PORT}"
            "/invocations"
        ),
        json=rqt_body,
    )

    if image_path != "":
        Path(image_path).unlink(missing_ok=True)

    if response.status_code != requests.codes.ok:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)

    model_response = response.json()
    predictions = model_response["predictions"][0]
    predictions.pop("product_id")

    result: list[PredictionResult] = [
        PredictionResult(category_id=int(category_id), probabilities=probabilities)
        for category_id, probabilities in zip(predictions.keys(), predictions.values())
    ]
    result.sort(key=lambda x: x.probabilities, reverse=True)

    if limit is not None:
        return result[:limit]
    return result
