from typing import Optional

from pydantic import BaseModel

class Product(BaseModel):
    designation: Optional[str] = ""
    description: Optional[str] = ""

