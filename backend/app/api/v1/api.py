from fastapi import APIRouter

from app.api.v1.endpoints import public, users, login, prediction

api_router = APIRouter()

api_router.include_router(public.router, tags=["public"])
api_router.include_router(login.router, tags=["login"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(prediction.router, prefix="/predict", tags=["prediction"])
