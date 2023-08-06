"""Route to check if the API is online."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api import dependencies
from app.core.settings import settings

router = APIRouter()


@router.get("/")
async def is_online(
    _db: Session = Depends(dependencies.get_db),
) -> str:
    """Check if the API is running and if DB session is active.

    Args:
        _db: active db session.

    Returns:
        Message saying the API is online.
    """
    return f"API {settings.PROJECT_NAME} is online!"
