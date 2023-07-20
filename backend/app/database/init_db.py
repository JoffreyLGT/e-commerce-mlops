from sqlalchemy.orm import Session

from app.crud.crud_user import user as crud_user
from app.schemas.user import UserCreate

from app.core.settings import settings
from app.database import base_class  # noqa: F401

# Make sure all SQL Alchemy models are imported (app.database.base) before initializing DB
# Otherwise, SQL Alchemy might fail to initialize relationships properly
# For more details: https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/28


def init_db(db: Session) -> None:
    # Tables are created with Alembic migrations

    user = crud_user.get_by_email(db, email=settings.FIRST_SUPERUSER)
    if not user:
        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = crud_user.create(db, obj_in=user_in)  # noqa: F841
