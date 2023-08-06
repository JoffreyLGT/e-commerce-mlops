"""Contains functions to manage authentication."""

from datetime import datetime, timedelta
from typing import Any

from jose import jwt
from passlib.context import CryptContext

from app.core.settings import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


ALGORITHM = "HS256"


def create_access_token(
    subject: str | Any,
    expires_delta: timedelta = timedelta(settings.ACCESS_TOKEN_EXPIRE_MINUTES),
) -> str:
    """Create a JWT access token.

    Args:
        subject: Information to store in the token.
        expires_delta: Validity time of the token. Defaults settings.ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        Encoded JWT token.
    """  # noqa: E501
    expire = datetime.utcnow() + expires_delta
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compare the plain password with the hashed password.

    Args:
        plain_password: password not hacked.
        hashed_password: hached password to compare.

    Returns:
        True if the hashed version of plain_password gives the same result as hashed_password.
    """  # noqa: E501
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash the password.

    Args:
        password: to hash.

    Returns:
        Hashed version of password.
    """
    return pwd_context.hash(password)
