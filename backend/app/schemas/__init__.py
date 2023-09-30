"""Import all schemas classes into the same file for import convenience."""

from app.schemas.prediction_feedback import PredictionFeedback as PredictionFeedback
from app.schemas.prediction_feedback import (
    PredictionFeedbackCreate as PredictionFeedbackCreate,
)
from app.schemas.prediction_feedback import (
    PredictionFeedbackInDB as PredictionFeedbackInDB,
)
from app.schemas.prediction_feedback import (
    PredictionFeedbackUpdate as PredictionFeedbackUpdate,
)
from app.schemas.prediction_result import PredictionResult as PredictionResult
from app.schemas.product import Product as Product
from app.schemas.token import Token as Token
from app.schemas.token import TokenPayload as TokenPayload
from app.schemas.user import User as User
from app.schemas.user import UserCreate as UserCreate
from app.schemas.user import UserInDB as UserInDB
from app.schemas.user import UserUpdate as UserUpdate
