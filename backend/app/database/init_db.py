"""Contains function to initiate Database when started."""
from sqlalchemy.orm import Session

from app.core.settings import settings
from app.crud.crud_user import user as crud_user

# Important to import base_class so it can be used when importing init_db
from app.database import base_class  # pyright: ignore  # noqa: F401
from app.schemas.user import UserCreate

# Make sure all SQLAlchemy models are imported (app.database.base) before using init_db
# Otherwise, SQL Alchemy might fail to initialize relationships properly
# For more details: https://github.com/tiangolo/full-stack-fastapi-postgresql/issues/28


def init_db(db: Session) -> None:
    """Initialise database and insert FIRST_ADMINUSER if it doesn't exists.

    All tables are created using alembic migrations.

    Args:
        db: session to connect to the DB.
    """
    if settings.FIRST_ADMINUSER is None or settings.FIRST_ADMINUSER_PASSWORD is None:
        print("FIRST_ADMINUSER env variable not defined: cannot add user to DB.")
        return

    user = crud_user.get_by_email(db, email=settings.FIRST_ADMINUSER)
    if not user:
        print("Adding FIRST_ADMINUSER to DB.")
        user_in = UserCreate(
            email=settings.FIRST_ADMINUSER,
            password=settings.FIRST_ADMINUSER_PASSWORD,
            is_admin=True,
        )
        user = crud_user.create(db, user_in=user_in)
    else:
        print("FIRST_ADMINUSER already in DB.")
