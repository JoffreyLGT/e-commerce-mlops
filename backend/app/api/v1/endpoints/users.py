"""Routes to manage users."""

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.encoders import jsonable_encoder
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import dependencies

router = APIRouter()


@router.get("/", response_model=list[schemas.User])
def read_users(
    db: Session = Depends(dependencies.get_db),
    skip: int = 0,
    limit: int = 100,
    # Authentication and access management, do not delete!
    _current_user: models.User = Depends(dependencies.get_current_active_admin),
) -> Any:
    """Retrieve users information."""
    users = crud.user.get_multi(db, skip=skip, limit=limit)
    return users


@router.post("/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(
    *,
    db: Session = Depends(dependencies.get_db),
    user_in: schemas.UserCreate,
    # Authentication and access management, do not delete!
    _current_user: models.User = Depends(dependencies.get_current_active_admin),
) -> Any:
    """Create a new user and return their information.

    Raises:
        HTTPException: 400 if user already exists in the system.

    Returns:
        New user information.
    """
    user = crud.user.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this username already exists in the system.",
        )
    user = crud.user.create(db, obj_in=user_in)
    return user


@router.put("/me", response_model=schemas.User)
def update_user_me(
    *,
    db: Session = Depends(dependencies.get_db),
    password: str = Body(None),
    email: EmailStr = Body(None),
    current_user: models.User = Depends(dependencies.get_current_active_user),
) -> Any:
    """Update user's own information.

    Returns:
        Updated user information.
    """
    current_user_data = jsonable_encoder(current_user)
    user_in = schemas.UserUpdate(**current_user_data)
    if password is not None:
        user_in.password = password
    if email is not None:
        user_in.email = email
    user = crud.user.update(db, db_obj=current_user, obj_in=user_in)
    return user


@router.get("/me", response_model=schemas.User)
def read_user_me(
    current_user: models.User = Depends(dependencies.get_current_active_user),
) -> Any:
    """Get current user information."""
    return current_user


@router.get("/{user_id}", response_model=schemas.User)
def read_user_by_id(
    user_id: int,
    # Authentication and access management, do not delete!
    _current_user: models.User = Depends(dependencies.get_current_active_admin),
    db: Session = Depends(dependencies.get_db),
) -> Any:
    """Get a specific user's information by their id.

    Raises:
        HTTPException: 404 if the user is not in db.

    Returns:
        User information.
    """
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system",
        )
    return user


@router.put("/{user_id}", response_model=schemas.User)
def update_user(
    *,
    db: Session = Depends(dependencies.get_db),
    user_id: int,
    user_in: schemas.UserUpdate,
    # Authentication and access management, do not delete!
    _current_user: models.User = Depends(dependencies.get_current_active_admin),
) -> Any:
    """Update a specific user's information by their id.

    Raises:
        HTTPException: 404 if the user is not in db.

    Returns:
        User information.
    """
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system",
        )
    user = crud.user.update(db, db_obj=user, obj_in=user_in)
    return user
