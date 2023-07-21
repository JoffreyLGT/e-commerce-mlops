from fastapi import APIRouter

from app.core.settings import settings


router = APIRouter()

@router.get("/")
async def is_online():
    return {f"API {settings.PROJECT_NAME} is online!"}