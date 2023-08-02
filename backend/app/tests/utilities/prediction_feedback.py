"""Utilities regarding prediction feedback management used in tests."""
import random

from numpy import product
from sqlalchemy.orm import Session

from app import crud
from app.schemas import PredictionFeedback, PredictionFeedbackCreate
from app.tests.utilities.utilities import random_category_id


def create_random_feedback(db: Session) -> PredictionFeedback:
    """Create a user with random information in DB.

    Args:
        db: database session.

    Returns:
        New user information.
    """

    feedback_in = PredictionFeedbackCreate(
        product_id=random.randint(0, 999999),
        real_category_id=random_category_id(),
        pred_category_id=random_category_id(),
        model_version="1.0",
    )
    feedback = crud.prediction_feedback.create(db=db, obj_in=feedback_in)
    return PredictionFeedback(
        id=feedback.id,
        product_id=feedback.product_id,
        pred_category_id=feedback.pred_category_id,
        real_category_id=feedback.real_category_id,
        model_version=feedback.model_version,
    )
