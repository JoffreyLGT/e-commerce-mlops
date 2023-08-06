"""Import all schemas classes into the same file for import convenience."""

from .prediction_feedback import (
    PredictionFeedback,
    PredictionFeedbackCreate,
    PredictionFeedbackInDB,
    PredictionFeedbackUpdate,
)
from .predictionresult import PredictionResult
from .product import Product
from .token import Token, TokenPayload
from .user import User, UserCreate, UserInDB, UserUpdate
