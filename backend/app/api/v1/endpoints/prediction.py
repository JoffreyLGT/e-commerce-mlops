from io import BytesIO
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, UploadFile, HTTPException
import numpy as np
from sqlalchemy.orm import Session
from PIL import Image

import app.api.dependencies as deps
from app import schemas, models
from datascience.classification import get_prediction

router = APIRouter()


@router.post("/prediction", response_model=List[schemas.PredictionResult])
async def predict_category(
    *,
    _db: Session = Depends(deps.get_db),
    designation: str = "",
    description: str = "",
    image: UploadFile | None = None,
    _current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    Predict the category of the product.
    """
    if image is None:
        image_data = np.zeros((254, 254, 3))
    else:
        extension = image.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if extension is False:
            raise HTTPException(
                400, detail="Invalid image extension. Must be jpg or jpeg."
            )
        image_data = Image.open(BytesIO(await image.read()))

    result = get_prediction(
        str(designation or ""), str(description or ""), np.asarray(image_data)
    )
    return [
        schemas.PredictionResult(prdtypecode=i[0], probabilities=i[1], label=i[2])
        for i in result
    ]
