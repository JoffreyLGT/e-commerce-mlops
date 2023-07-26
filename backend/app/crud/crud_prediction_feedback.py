"""Functions to manage PredictionFeedback entries in DB.
"""
from sqlalchemy.orm import Session
from typing import List

from app.core.settings import settings
from app.crud.base import CRUDBase
from app.models.prediction_feedback import PredictionFeedback
from app.schemas.prediction_feedback import (
    PredictionFeedbackCreate,
    PredictionFeedbackUpdate,
)


class CRUDPredictionFeedback(
    CRUDBase[PredictionFeedback, PredictionFeedbackCreate, PredictionFeedbackUpdate]
):
    """Manage PredictionFeedback entries in DB.

    Args:
        CRUDBase: generic methods to manage entries in DB.
    """

    def get_by_product_id(
        self, db: Session, *, product_id: int, model_version: str | None
    ) -> PredictionFeedback | None:
        """Fetch the entry for a specific product id from DB.

        Args:
            db: _description_
            product_id: product identifier in Rakuten systems
            model_version: version of the model used for prediction

        Returns:
            Corresponding entry or None if there is no match in DB.
        """
        version = model_version or settings.MODEL_VERSION
        return (
            db.query(PredictionFeedback)
            .filter(
                PredictionFeedback.product_id == product_id,
                PredictionFeedback.model_version == version,
            )
            .first()
        )

    def get_by_model_version(
        self,
        db: Session,
        *,
        model_version: str | None = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[PredictionFeedback]:
        """Fetch all entries for a model version.

        Args:
            db: _description_
            product_id: product identifier in Rakuten systems
            model_version: version of the model used for prediction

        Returns:
            Corresponding entry or None if there is no match in DB.
        """
        if model_version is None:
            return db.query(PredictionFeedback).offset(skip).limit(limit).all()
        return (
            db.query(PredictionFeedback)
            .filter(
                PredictionFeedback.model_version == model_version,
            )
            .offset(skip)
            .limit(limit)
            .all()
        )


prediction_feedback = CRUDPredictionFeedback(PredictionFeedback)
