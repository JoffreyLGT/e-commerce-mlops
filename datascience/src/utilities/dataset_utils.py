"""Functions to generate Datasets."""


import logging
from pathlib import Path

import pandas as pd
from pydantic.types import DirectoryPath

from src.core.custom_errors import MissingDataError, MissingEnvironmentVariableError

logger = logging.getLogger(__name__)


def load_dataset(datadir: DirectoryPath) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset and return a DataFrame with both features and target.

    Args:
        datadir: Directory where data are located. Defaults to settings.ORIGINAL_DATA_DIR.

    Returns:
        (features, target) with the content of the dataset.
    """  # noqa: E501
    return (
        pd.read_csv(f"{datadir}/X.csv", index_col=0),
        pd.read_csv(f"{datadir}/y.csv", index_col=0),
    )


def get_dataset_missing_files(path: str) -> list[str]:
    """Check dataset directory content for missing files.

    We are just checking if the files are in the directory provided, not their content.
    The files we are looking for are:
    - X.csv
    - y.csv
    - images directory.
    """
    missing_files: list[str] = []
    if not Path(f"{path}/X.csv").is_file():
        missing_files.append("X.csv")
    if not Path(f"{path}/y.csv").is_file():
        missing_files.append("y.csv")
    if not Path(f"{path}/images").is_dir():
        missing_files.append("images")
    return missing_files


def get_remaining_dataset_path() -> Path:
    """Return the path of remaining dataset.

    Since remaining dataset directory is empty when starting,
    this function will check its content and return ORIGINAL_DATA_DIR
    instead, but only if it has files.
    Note: since we are creating REMAINING_DATA_DIR in settings,
    MissingEnvironmentVariableError should never be raised.

    Raises:
        MissingEnvironmentVariableError: if REMAINING_DATA_DIR is undefined.
        MissingDataError: if REMAINING_DATA_DIR and ORIGINAL_DATA_DIR are missing data.

    Returns:
        Path to the remaining Dataset.
    """
    # Must be placed here to avoid circular import errors
    from src.core.settings import get_dataset_settings

    settings = get_dataset_settings()
    if settings.REMAINING_DATA_DIR is None:
        raise MissingEnvironmentVariableError("REMAINING_DATA_DIR")

    # Create the directory if it doesn't exist
    Path(settings.REMAINING_DATA_DIR).mkdir(parents=True, exist_ok=True)
    if len(get_dataset_missing_files(settings.REMAINING_DATA_DIR)) > 0:
        if (
            settings.ORIGINAL_DATA_DIR is not None
            and len(get_dataset_missing_files(str(settings.ORIGINAL_DATA_DIR))) == 0
        ):
            logger.info(
                "REMAINING_DATA_DIR has missing files. Use ORIGINAL_DATA_DIR instead."
            )
            return Path(settings.ORIGINAL_DATA_DIR)

        raise MissingDataError(  # noqa: TRY003
            f"Data missing in REMAINING_DATA_DIR ({settings.REMAINING_DATA_DIR}) and ORIGINAL_DATA_DIR ({settings.ORIGINAL_DATA_DIR})."  # noqa: E501
        )

    return Path(settings.REMAINING_DATA_DIR)
