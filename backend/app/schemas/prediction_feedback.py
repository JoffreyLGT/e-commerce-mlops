"""
Define Pydantic schemas for a PredictionFeedback object.
"""


from pydantic import BaseModel, validator

from app.core.security import settings


class PredictionFeedbackBase(BaseModel):
    """
    Shared attributes between each class.
    """

    product_id: int
    model_version: str | None = "1.0"
    real_category_id: int
    pred_category_id: int


class PredictionFeedbackCreate(PredictionFeedbackBase):
    """
    Properties to receive via API on creation.
    """

    product_id: int
    real_category_id: int
    pred_category_id: int


class PredictionFeedbackUpdate(PredictionFeedbackBase):
    """
    Properties to receive via API on update.
    """


class PredictionFeedbackInDBBase(PredictionFeedbackBase):
    """
    Base properties stored in DB. They are the one needed for all entries.
    """

    id: int
    model_version: str | None = None
    product_id: int
    real_category_id: int
    pred_category_id: int

    class Config:  # pylint: disable=R0903
        """
        Pydantic configuration.

        Attributes:
            orm_mode            Map the models to ORM objects.
            validate_assignment Perform validation on assignment to attributes.
        """

        orm_mode = True
        validate_assignment = True

    @validator("model_version")
    def set_model_version(cls, model_version):  # pylint: disable=E0213
        """Assign a default value to model_version if set to None."""
        return model_version or settings.MODEL_VERSION


class PredictionFeedback(PredictionFeedbackInDBBase):
    """
    Properties to return to client. By default, all the base properties stored in DB.
    """


class PredictionFeedbackInDB(PredictionFeedbackInDBBase):
    """
    Additional properties stored in DB. Most of the time, they are optionals.
    """
