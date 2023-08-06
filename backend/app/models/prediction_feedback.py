"""Definition of a prediction feedback object.

Created and stored when our clients are sending us what
product category their seller chose and what we predicted.
"""

from sqlalchemy import Integer, String
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base_class import Base


class PredictionFeedback(Base):
    """Contains the feedback we got from our client regarding our prediction."""

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Indicate table name to SQLAlchemy."""
        return "prediction_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    product_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False, index=True)

    real_category_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    pred_category_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
