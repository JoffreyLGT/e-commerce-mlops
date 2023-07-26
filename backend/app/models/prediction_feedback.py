"""
Definition of a prediction feedback object.
Created and stored when our clients are sending us what
product category their seller chose and what we predicted.
"""

from sqlalchemy import Column, Integer, String

from app.database.base_class import Base


# pylint: disable=R0903
class PredictionFeedback(Base):
    """
    Contains the feedback we got from our client regarding our prediction.
    """

    __tablename__ = "prediction_feedback"  # type: ignore

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False, index=True)
    model_version = Column(String, nullable=False, index=True)

    real_category_id = Column(Integer, nullable=False, index=True)
    pred_category_id = Column(Integer, nullable=False, index=True)
