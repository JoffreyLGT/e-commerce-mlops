"""Routes to manage prediction feedbacks from Rakuten."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import dependencies
from app.core.settings import settings

router = APIRouter()


@router.get("/", response_model=list[schemas.PredictionFeedback])
def get_feedbacks(
    db: Session = Depends(dependencies.get_db),
    # Authentication and access management, do not delete!
    _current_user: models.User = Depends(dependencies.get_current_active_admin),
    model_version: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """Retrieve prediction feedbacks from DB."""
    feedbacks = crud.prediction_feedback.get_by_model_version(
        db, model_version=model_version, skip=skip, limit=limit
    )
    return feedbacks


@router.post(
    "/", response_model=schemas.PredictionFeedback, status_code=status.HTTP_201_CREATED
)
def create_feedback(
    *,
    db: Session = Depends(dependencies.get_db),
    # Authentication and access management, do not delete!
    _current_user: models.User = Depends(dependencies.get_current_active_user),
    feedback_in: schemas.PredictionFeedbackCreate,
) -> Any:
    """Create new prediction feedback."""
    feedback = crud.prediction_feedback.get_by_product_id(
        db, product_id=feedback_in.product_id, model_version=settings.MODEL_VERSION
    )
    if feedback:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A prediction feedback with this product id already exists in the system.",  # noqa: E501
        )
    feedback = crud.prediction_feedback.create(db, obj_in=feedback_in)
    return feedback
