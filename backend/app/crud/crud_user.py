"""Functions to manage Users entries in DB."""

from typing import Any

from sqlalchemy.orm import Session

from app.core.security import get_password_hash, verify_password
from app.crud.base import CRUDBase
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    """Manage User entries in DB.

    Args:
        CRUDBase: generic methods to manage entries in DB.
    """

    def get_by_email(self, db: Session, *, email: str) -> User | None:
        """Get User from DB by their email.

        Args:
            db: session to run requests.
            email: of the user.

        Returns:
            User if found, None otherwise.
        """
        return db.query(User).filter(User.email == email).first()

    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        """Create a new user in DB.

        Args:
            db: session to run requests.
            obj_in: User to add in DB.

        Returns:
            User created in DB, so with their extra properties.
        """
        db_user = User(
            email=obj_in.email,
            hashed_password=get_password_hash(obj_in.password),
            is_admin=obj_in.is_admin,
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    def update(
        self, db: Session, *, db_obj: User, obj_in: UserUpdate | dict[str, Any]
    ) -> User:
        """Update user in DB.

        Args:
            db: session to run requests.
            db_obj: user to edit in DB.
            obj_in: user with updated information.

        Returns:
            User with updated information.
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if update_data.get("password", None):
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        return super().update(db, db_obj=db_obj, obj_in=update_data)

    def authenticate(self, db: Session, *, email: str, password: str) -> User | None:
        """Authenticate User with their credentials.

        Args:
            db: session to run requests.
            email: user login.
            password: user password.

        Returns:
            User information if the authenticated, None otherwise.
        """
        user = self.get_by_email(db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def is_active(self, user: User) -> bool:
        """Check if a User account is active and can use the API.

        Args:
            user: to check.

        Returns:
            True if User account is active.
        """
        return user.is_active

    def is_admin(self, user: User) -> bool:
        """Check if a User account has admin privilege.

        Args:
            user: to check.

        Returns:
            True if User account has admin privilege.
        """
        return user.is_admin


user = CRUDUser(User)
