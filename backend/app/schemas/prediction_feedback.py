"""Define Pydantic schemas for a PredictionFeedback object."""


import json

from pydantic import BaseModel, validator

from app.core.security import settings


class PredictionFeedbackBase(BaseModel):
    """Shared attributes between each class."""

    product_id: int
    model_version: str | None = "1.0"
    real_category_id: int
    pred_category_id: int

    def to_json_str(self) -> str:
        """Export object to a string with JSON format.

        Returns:
            Object in a string with JSON format.
        """
        return json.dumps(self.__dict__)

    def to_json(self):
        """Export object to JSON format.

        Returns:
            Object in JSON format.
        """
        return json.loads(self.to_json_str())

    @classmethod
    def from_json(cls, json_str):
        """Load object from a JSON.

        Args:
            json_str: JSON of the object.

        Returns:
            A PredictionFeedbackInDBBase with the information from json_str.
        """
        json_dict = json.loads(json_str)
        return cls(**json_dict)


class PredictionFeedbackCreate(PredictionFeedbackBase):
    """Properties to receive via API on creation."""

    product_id: int
    real_category_id: int
    pred_category_id: int


class PredictionFeedbackUpdate(PredictionFeedbackBase):
    """Properties to receive via API on update."""


class PredictionFeedbackInDBBase(PredictionFeedbackBase):
    """Base properties stored in DB. They are the one needed for all entries."""

    id: int
    model_version: str | None = None
    product_id: int
    real_category_id: int
    pred_category_id: int

    @validator("model_version")
    def set_model_version(cls, model_version):
        """Assign a default value to model_version if set to None."""
        return model_version or settings.MODEL_VERSION

    class Config:
        """Pydantic configuration.

        Attributes:
            orm_mode            Map the models to ORM objects.
            validate_assignment Perform validation on assignment to attributes.
        """

        orm_mode = True
        validate_assignment = True


class PredictionFeedback(PredictionFeedbackInDBBase):
    """Properties to return to client.

    By default, all the base properties stored in DB.
    """


class PredictionFeedbackInDB(PredictionFeedbackInDBBase):
    """Additional properties stored in DB. Most of the time, they are optionals."""
