"""Contains all the settings needed to run the API.

Most of them are coming from env variables, others are
constructed or simply defined.
Note: important variable must have validator using pydantic.
"""

import secrets
from typing import Any, Dict, List, Union

from pydantic import AnyHttpUrl, BaseSettings, EmailStr, PostgresDsn, validator


class Settings(BaseSettings):
    """Settings to use in the API. Values are overwritten by environment variables."""

    # pylint: disable=E0213
    DOMAIN: str | None = None

    PROJECT_NAME: str | None = None
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    SERVER_NAME: str | None = None
    SERVER_HOST: AnyHttpUrl | None = None

    POSTGRES_SERVER: str | None = None
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_DB: str | None = None
    SQLALCHEMY_DATABASE_URI: PostgresDsn | None = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: str | None, values: Dict[str, Any]) -> Any:
        """Ensure we have a valid DB connection.

        Validate the information provided through env variable to ensure we are able to
        build a correct PostgreSQL connection string if SQLALCHEMY_DATABASE_URI is
        not provided.
        """
        # Check if SQLALCHEMY_DATABASE_URI was provided through env
        if isinstance(v, str):
            return v

        # Check if we have all the information to build Postgre URI
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=f"{values.get('POSTGRES_SERVER') or ''}",
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:8000"]
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Validate the format of BACKEND_CORS_ORIGINS.

        Raises:
            ValueError: if value is not a string or a list of string.

        Returns:
            Single string with all CORS separated by commas.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        if isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    TEST_USER: EmailStr = EmailStr("user@test.com")
    FIRST_ADMINUSER: EmailStr | None = None
    FIRST_ADMINUSER_PASSWORD: str | None = None
    USERS_OPEN_REGISTRATION: bool = False

    RESET_TOKEN_EXPIRE_HOURS: int = 48

    # FIXME set dynamically to not update codebase when updating models
    MODEL_VERSION: str = "1.0"

    # pylint: disable=R0903
    class Config:
        """Pydantic configuration."""

        case_sensitive = True


settings = Settings()
