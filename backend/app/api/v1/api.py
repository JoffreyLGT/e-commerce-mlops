"""Create an APIRouter object with all endpoints."""

from fastapi import APIRouter

from app.api.v1.endpoints import login, prediction, prediction_feedback, public, users

api_router = APIRouter()

api_router.include_router(public.router, tags=["Public"])
api_router.include_router(login.router, tags=["Logins"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(prediction.router, prefix="/predict", tags=["Predictions"])
api_router.include_router(
    prediction_feedback.router, prefix="/feedback", tags=["Feedbacks"]
)
