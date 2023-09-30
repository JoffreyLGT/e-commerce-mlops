"""Contains all the settings needed in datascience project.

Keep in mind classes inheriting from BaseSettings can have their properties
replaced by environment variables. To add settings that cannot be changed,
add them in settings.py.
Note: important variable must have a validator using pydantic.
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, BaseSettings, DirectoryPath, validator

from src.core import constants
from src.utilities.dataset_utils import ensure_dataset_dir_content


class _CommonSettings(BaseSettings):
    """Common settings of the project."""

    TARGET_ENV: Literal["development", "staging", "production"] = "development"
    CONSOLE_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    FILE_LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOGS_DIR: str = "logs"

    @validator("LOGS_DIR")
    def create_log_dir(
        cls,  # noqa: N805 false positive, first argument must be cls for validator
        path: str,
    ) -> DirectoryPath:
        """Ensure the directory contains the necessary values."""
        full_path: Path = (
            Path(path) if Path(path).is_absolute() else Path(constants.ROOT_DIR) / path
        )
        full_path.mkdir(parents=True, exist_ok=True)
        return full_path

    LOGS_FILE_NAME: str = "datascience.log.json"

    # Defined as ENV variable on Github Actions. Used for specific conditions.
    IS_GH_ACTION: bool = False

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


def check_has_extension(file_name: str, extension: str) -> str:
    """Ensure the file has the extension .png."""
    logger = logging.getLogger(__file__)
    if Path(file_name).suffix != extension:
        final_name = f"{file_name}.{extension}"
        logger.warning(
            f"{file_name} doesn't have {extension} extension. "
            f"Therefore, final name will be {final_name}"
        )
        return final_name
    return file_name


class _TrainingSettings(_CommonSettings):
    """Settings used in training scripts."""

    TEXT_IDFS_FILE_NAME: str = "text_idfs.json"

    @validator("TEXT_IDFS_FILE_NAME")
    def check_has_json_extension_1(cls, name: str) -> str:  # noqa: N805
        """Ensure the file has .json extension."""
        return check_has_extension(name, ".json")

    TEXT_VOCABULARY_FILE_NAME: str = "text_vocabulary.json"

    @validator("TEXT_VOCABULARY_FILE_NAME")
    def check_has_json_extension(cls, name: str) -> str:  # noqa: N805
        """Ensure the file has .json extension."""
        return check_has_extension(name, ".json")

    TRAINING_HISTORY_FILE_NAME: str = "training_history.png"

    @validator("TRAINING_HISTORY_FILE_NAME")
    def check_has_png_extension(cls, name: str) -> str:  # noqa: N805
        """Ensure the file has .png extension."""
        return check_has_extension(name, ".png")

    CLASSIFICATION_REPORT_FILE_NAME: str = "classifaction_report.txt"

    @validator("CLASSIFICATION_REPORT_FILE_NAME")
    def check_has_txt_extension(cls, name: str) -> str:  # noqa: N805
        """Ensure the file has .txt extension."""
        return check_has_extension(name, ".txt")

    CONFUSION_MATRIX_FILE_NAME: str = "confusion_matrix.png"

    @validator("CONFUSION_MATRIX_FILE_NAME")
    def check_has_png_extension_2(cls, name: str) -> str:  # noqa: N805
        """Ensure the file has .png extension."""
        return check_has_extension(name, ".png")

    REQUIREMENTS_FILE_NAME: str = "requirements.txt"

    @validator("REQUIREMENTS_FILE_NAME")
    def check_has_txt_extension_2(cls, name: str) -> str:  # noqa: N805
        """Ensure the file has .txt extension."""
        return check_has_extension(name, ".txt")

    MLFLOW_REGISTRY_URI: str = "mlruns"

    @validator("MLFLOW_REGISTRY_URI")
    def is_file_uri(cls, store_uri: str) -> str:  # noqa: N805
        """Ensure the string has file URI format."""
        return Path(store_uri).absolute().as_posix()

    MLFLOW_TRACKING_URI: str = "mlruns"

    @validator("MLFLOW_TRACKING_URI")
    def is_file_uri_2(cls, store_uri: str) -> str:  # noqa: N805
        """Ensure the string has file URI format."""
        return Path(store_uri).absolute().as_posix()

    MLFLOW_SET_DEFAULT_MODELS: bool = True


class _DatasetSettings(BaseSettings):
    """Settings to manage datasets."""

    # Number of threads to use when doing images conversion in create_datasets.py
    IMG_PROCESSING_NB_THREAD: int = 4

    # Directory where data used to train the models are stored.
    ORIGINAL_DATA_DIR: DirectoryPath | None = None

    @validator("ORIGINAL_DATA_DIR")
    def must_contain_data(
        cls,  # noqa: N805 false positive, first argument must be cls for validator
        path: DirectoryPath | None,
    ) -> str | None:
        """Ensure the directory contains the necessary values."""
        return ensure_dataset_dir_content(path=path, root_dir=Path(constants.ROOT_DIR))

    # Directory where remaining data not already in a dataset are stored.
    REMAINING_DATA_DIR: str | None = None

    @validator("REMAINING_DATA_DIR")
    def create_if_possible(
        cls,  # noqa: N805 false positive, first argument must be cls for validator
        path: str | None,
        values: dict[str, Any],
    ) -> str | None:
        """Ensure the directory is created."""
        logger = logging.getLogger(__name__)
        full_path: str
        if path is None:
            if values["ORIGINAL_DATA_DIR"] is None:
                logger.warning(
                    "REMAINING_DATA_DIR and ORIGINAL_DATA_DIR are not defined."
                )
                return None

            parent = Path(values["ORIGINAL_DATA_DIR"]).parent
            full_path = str(Path(parent) / "_remaining")
            logger.info(
                f"REMAINING_DATA_DIR not defined, create it in the parent folder of ORIGINAL_DATA_DIR: {full_path}."  # noqa: E501
            )
        else:
            full_path = path
        Path(full_path).mkdir(parents=True, exist_ok=True)
        return full_path

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


class _MobileNetImageModelSettings(BaseModel):
    """Define constants regarding the MobileNet image model."""

    # Model images characteristics
    IMG_WIDTH: int = 224
    IMG_HEIGHT: int = 224

    # Optimized images settings
    IMG_KEEP_RATIO: bool = True
    IMG_GRAYSCALED: bool = False


@lru_cache(maxsize=1)
def get_common_settings() -> _CommonSettings:
    """Return the common settings.

    @lru_cache(maxsize=1) ensure we create only one instance, cache it,
    and send it back everytime the function is called.
    """
    return _CommonSettings()


@lru_cache(maxsize=1)
def get_dataset_settings() -> _DatasetSettings:
    """Return the dataset settings.

    @lru_cache(maxsize=1) ensure we create only one instance, cache it,
    and send it back everytime the function is called.
    """
    return _DatasetSettings()


@lru_cache(maxsize=1)
def get_mobilenet_image_model_settings() -> _MobileNetImageModelSettings:
    """Return the MobileNet image model settings.

    @lru_cache(maxsize=1) ensure we create only one instance, cache it,
    and send it back everytime the function is called.
    """
    return _MobileNetImageModelSettings()


@lru_cache(maxsize=1)
def get_training_settings() -> _TrainingSettings:
    """Return the training settings.

    @lru_cache(maxsize=1) ensure we create only one instance, cache it,
    and send it back everytime the function is called.
    """
    return _TrainingSettings()
