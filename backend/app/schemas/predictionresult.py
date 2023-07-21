from pydantic import BaseModel


class PredictionResult(BaseModel):
    prdtypecode: int
    probabilities: float
    label: str
