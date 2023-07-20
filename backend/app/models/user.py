from typing import TYPE_CHECKING, List

from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.orm import relationship

from app.database.base_class import Base

class User(Base):
    id: int = Column(Integer, primary_key=True, index=True)
    email: str = Column(String, unique=True, index=True, nullable=False)
    hashed_password: str = Column(String, nullable=False)
    is_active: bool = Column(Boolean(), default=True)
    is_admin: bool = Column(Boolean(), default=False)
