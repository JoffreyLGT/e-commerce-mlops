"""Define Pydantic schemas for a Product object."""
from pydantic import BaseModel


class Product(BaseModel):
    """Product to be sold on a platform."""

    designation: str | None = ""
    description: str | None = ""
