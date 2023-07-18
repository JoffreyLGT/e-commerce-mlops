import secrets
from typing import List, Union

from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str   
    API_V1_STR: str = "/api/v1"


settings = Settings()
