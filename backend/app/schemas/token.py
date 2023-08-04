"""Define Pydantic schemas for a token object."""
from pydantic import BaseModel


class Token(BaseModel):
    """Attributes of a JWT token."""

    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    """Payload of a JWT token."""

    sub: int | None = None
