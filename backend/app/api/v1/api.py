from fastapi import APIRouter

from app.api.v1.endpoints import public

api_router = APIRouter()

api_router.include_router(public.router, tags=["public"])