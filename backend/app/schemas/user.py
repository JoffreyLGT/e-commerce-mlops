"""Define Pydantic schemas for a User object."""
from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """Shared attributes between each class."""

    email: EmailStr | None = None
    is_active: bool | None = True
    is_admin: bool = False


class UserCreate(UserBase):
    """Properties to receive via API on creation."""

    email: EmailStr
    password: str


class UserUpdate(UserBase):
    """Properties to receive via API on update."""

    password: str


class UserInDBBase(UserBase):
    """Base properties stored in DB. They are the one needed for all entries."""

    id: int

    class Config:
        """Pydantic configuration.

        Attributes:
            orm_mode            Map the models to ORM objects.
        """

        orm_mode = True


class User(UserInDBBase):
    """Properties to return to client.

    By default, all the base properties stored in DB.
    """


# Additional properties stored in DB
class UserInDB(UserInDBBase):
    """Additional properties stored in DB. Most of the time, they are optionals."""

    hashed_password: str
