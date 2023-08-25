"""Contains all metadata used by FastAPI."""

from typing import Any, TypedDict

from app.core.settings import settings


class FastAPIMetadata(TypedDict):
    """Definition of the app metadata."""

    title: str
    summary: str
    version: str
    contact: Any
    openapi_url: str


title = f"{settings.PROJECT_NAME} API"

app_metadata: FastAPIMetadata = {
    "title": title,
    "summary": f"{title} predicts the category of your products.",
    "version": "1.0.0",
    "contact": {
        "name": "Mai23 MLOPS E-commerce team",
        "url": "https://github.com/JoffreyLGT/e-commerce-mlops",
    },
    "openapi_url": f"{settings.API_V1_STR}/openapi.json",
}

# ruff: noqa: E501
tags_metadata = [
    {
        "name": "Public",
        "description": "Test if the API is online and functional!",
    },
    {
        "name": "Logins",
        "description": "Get JWT access token or test the validity of an existing one.",
    },
    {
        "name": "Predictions",
        "description": "Get the category of your product of the probabilities to be in each category.",
    },
    {
        "name": "Feedbacks",
        "description": "Send back our prediction and the category selected by your customer so we can monitor the model accuracy.",
    },
    {
        "name": "Users",
        "description": "Operations with users.",
    },
]
