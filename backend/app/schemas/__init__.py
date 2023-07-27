"""
Import all schemas classes into the same file for import convenience.
"""

from .token import Token, TokenPayload
from .user import User, UserCreate, UserInDB, UserUpdate
from .product import Product
from .predictionresult import PredictionResult
from .prediction_feedback import (
    PredictionFeedback,
    PredictionFeedbackCreate,
    PredictionFeedbackInDB,
    PredictionFeedbackUpdate,
)
