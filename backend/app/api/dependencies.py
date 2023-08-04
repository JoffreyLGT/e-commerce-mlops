"""Contains dependencies used by the routes."""
from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from jose.exceptions import JWTError
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.core import security
from app.core.settings import settings
from app.database.session import SessionLocal

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)


def get_db() -> Generator:
    """Initiate a session with DB.

    Yields:
        Generator object containing the db session.
    """
    db = None
    try:
        db = SessionLocal()
        yield db
    finally:
        if db is not None:
            db.close()


def get_current_user(
    db: Session = Depends(get_db), token: str = Depends(reusable_oauth2)
) -> models.User:
    """Get the current user information.

    Args:
        db: database session. Defaults to Depends(get_db).
        token: user identication Bearer token. Defaults to Depends(reusable_oauth2).

    Raises:
        HTTPException: 403 if credentials are invalid.
        HTTPException: 403 if user is not found.

    Returns:
        Current user information.
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[security.ALGORITHM]
        )
        token_data = schemas.TokenPayload(**payload)
    except (JWTError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        ) from exc
    user = crud.user.get(db, id=token_data.sub)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User not found"
        )
    return user


def get_current_active_user(
    current_user: models.User = Depends(get_current_user),
) -> models.User:
    """Get current user information if user is active.

    Args:
        current_user: authenticate user and fetch its data from DB.

    Raises:
        HTTPException: 403 if user is inactive.

    Returns:
        User information.
    """
    if not crud.user.is_active(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user"
        )
    return current_user


def get_current_active_admin(
    current_user: models.User = Depends(get_current_active_user),
) -> models.User:
    """Get current user information if user is active and admin.

    Args:
        current_user: authenticate user, fetch its data from DB and check if active.

    Raises:
        HTTPException: 403 if user is inactive.

    Returns:
        User information.
    """
    if not crud.user.is_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges",
        )
    return current_user
