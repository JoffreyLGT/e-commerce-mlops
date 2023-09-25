"""Utilities to manage MLFlow."""


import mlflow
from mlflow.models import model

from src.core import constants, settings


def setup_mlflow(experiment_name: str, tags: dict[str, str]) -> None:
    """Setup MLFlow uris and experiments.

    Args:
        experiment_name: name of the experiment.
        tags: to add to the experiment.
    """
    mlflow.set_registry_uri(settings.get_training_settings().MLFLOW_REGISTRY_URI)
    mlflow.set_tracking_uri(settings.get_training_settings().MLFLOW_TRACKING_URI)

    register_default_models()

    mlflow.set_experiment(experiment_name)
    mlflow.set_tags(tags)


def register_default_models() -> list[str]:
    """Register default models in MLFlow.

    Edit constants module to add, remove or edit models.
    """
    mlflow.set_tracking_uri(settings.get_training_settings().MLFLOW_TRACKING_URI)

    client = mlflow.MlflowClient()
    created_models = list()
    for model_configs in constants.MLFLOW_DEFAULT_MODELS.items():
        _, model_info = model_configs
        name = model_info["name"]
        if len(client.search_registered_models(f"name = '{name}'")) == 0:
            results = client.create_registered_model(
                name=model_info["name"],
                description=model_info["description"],
                tags=model_info["tags"],
            )
            created_models.append(results)
    return created_models


def set_staging_stage(
    model_info: model.ModelInfo,
    mlflow_model_name: str,
    tags: dict[str, str] | None = None,
) -> None:
    """Set model to staging stage in MLFlow.

    Args:
        model_info: information regarding the model.
        mlflow_model_name: registered model name on MLFlow.
        tags: to add to the registered model.
    """
    client = mlflow.MlflowClient()
    if len(client.search_registered_models(f"name = '{mlflow_model_name}'")) == 0:
        client.create_registered_model(name=mlflow_model_name, tags=tags)

    mv = client.create_model_version(
        name=mlflow_model_name,
        source=model_info.model_uri,
        run_id=model_info.run_id,
        tags=tags,
    )

    client.transition_model_version_stage(
        name=mlflow_model_name,
        version=mv.version,
        stage="staging",
        archive_existing_versions=True,
    )
