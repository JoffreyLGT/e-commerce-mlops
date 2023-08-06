"""Base class to be imported in all DB classes."""

from typing import Any

from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base of DB classes with default attributes."""

    id: Any
    __name__: str

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate __tablename__ automatically based on __name__."""
        return cls.__name__.lower()
