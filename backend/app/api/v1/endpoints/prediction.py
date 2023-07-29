from io import BytesIO
from typing import Any, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

import app.api.dependencies as deps
from app import models, schemas
from datascience.classification import get_prediction

router = APIRouter()


@router.post("/", response_model=List[schemas.PredictionResult])
async def predict_category(
    *,
    designation: str = "",
    description: str = "",
    image: UploadFile | None = None,
    limit: int | None = None,
    _current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Predict the category of the product.
    """
    if image is None:
        image_data = np.zeros((254, 254, 3))
    else:
        extension = (
            False
            if image.filename is None
            else image.filename.split(".")[-1] in ("jpg", "jpeg")
        )
        if extension is False:
            raise HTTPException(
                400,
                detail="Invalid image extension. Image must be in JPEG or JPG format.",
            )
        try:
            # Open the image with PIL
            image_data = Image.open(BytesIO(await image.read()))
        except UnidentifiedImageError as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Image must be in JPEG or JPG format.",
            ) from exc

    if limit is not None and limit <= 0:
        raise HTTPException(
            400,
            detail="Invalid value for limit. Must be a positive integer.",
        )

    result = get_prediction(
        designation,
        description,
        np.asarray(image_data),
        limit,
    )
    return [
        schemas.PredictionResult(prdtypecode=i[0], probabilities=i[1], label=i[2])
        for i in result
    ]
