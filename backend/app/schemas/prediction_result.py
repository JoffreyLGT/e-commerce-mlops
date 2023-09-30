"""Define Pydantic schemas for a PredictionResult object."""
from pydantic import BaseModel


class PredictionResult(BaseModel):
    """Probabilities for a product to be in a category."""

    category_id: int
    probabilities: float
