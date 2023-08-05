"""Base CRUD class containing Create, Read, Update, Delete functions."""

from typing import Any, Generic, TypeVar

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.base_class import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """CRUD object with default methods to Create, Read, Update, Delete (CRUD).

    Args:
        Generic: array with types.
    """

    def __init__(self, model: type[ModelType]):
        """Create a new CRUD object.

        Args:
            model: SQLAlchemy model class.
        """
        self.model = model

    def get(self, db: Session, id: Any) -> ModelType | None:
        """Fetch an entry from DB.

        Args:
            db: session to run requests.
            id: of the entry.

        Returns:
            First entry with the provided id.
        """
        return db.query(self.model).filter(self.model.id == id).first()

    def get_multi(
        self, db: Session, *, skip: int = 0, limit: int = 100
    ) -> list[ModelType]:
        """Fetch multiple entries from DB.

        Args:
            db: session to run requests.
            skip: number of entries to skip. Defaults to 0.
            limit: number of entries to return. Defaults to 100.

        Returns:
            List of entries.
        """
        return db.query(self.model).offset(skip).limit(limit).all()

    def create(self, db: Session, *, obj_in: CreateSchemaType) -> ModelType:
        """Create a new entry in DB.

        Args:
            db: session to run requests.
            obj_in: Entry to insert.

        Returns:
            Entry added in DB.
        """
        obj_in_data = jsonable_encoder(obj_in)
        db_obj = self.model(**obj_in_data)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        *,
        db_obj: ModelType,
        obj_in: UpdateSchemaType | dict[str, Any]
    ) -> ModelType:
        """Update entry in DB.

        Args:
            db: session to run requests.
            db_obj: Entry from DB.
            obj_in: Entry with updated information.

        Returns:
            Updated entry from DB.
        """
        obj_data = jsonable_encoder(db_obj)
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def remove(self, db: Session, *, id: int) -> ModelType:
        """Remove entry from DB.

        Args:
            db: session to run requests.
            id: of the entry to remove.

        Returns:
            Deleted entry.
        """
        obj = db.get(self.model, id)  # pyright: ignore
        assert obj is not None
        db.delete(obj)
        db.commit()
        return obj
