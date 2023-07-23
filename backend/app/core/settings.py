"""
Contains all the settings needed to run the API.
Most of them are coming from env variables, others are
constructed or simply defined.
Note: important variable must have validator using pydantic.
"""

import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import AnyHttpUrl, EmailStr, PostgresDsn, validator, BaseSettings


class Settings(BaseSettings):
    # pylint: disable=E0213
    DOMAIN: str

    PROJECT_NAME: str
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    SERVER_NAME: str
    SERVER_HOST: AnyHttpUrl

    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        """
        Validate the information provided through env variable to ensure
        we are able to build a correct PostgreSQL connection string.
        """
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:8000"]
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """
        Validate the format of BACKEND_CORS_ORIGINS and ensure we are sending
        back a list of strings.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        if isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    FIRST_ADMINUSER: EmailStr
    FIRST_ADMINUSER_PASSWORD: str
    USERS_OPEN_REGISTRATION: bool = False

    RESET_TOKEN_EXPIRE_HOURS: int = 48

    # pylint: disable=C0115+R0903
    class Config:
        case_sensitive = True


settings = Settings()
