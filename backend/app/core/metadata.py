"""Contains all metadata used by FastAPI."""


from app.core.settings import settings

title = f"{settings.PROJECT_NAME} API"
# description = ""

app_metadata = {
    "title": title,
    # "description": description,
    "summary": f"{title} predicts the category of your products.",
    "version": "1.0.0",
    "contact": {
        "name": "Mai23 MLOPS E-commerce team",
        "url": "https://github.com/JoffreyLGT/e-commerce-mlops",
    },
    "openapi_url": f"{settings.API_V1_STR}/openapi.json",
}

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
