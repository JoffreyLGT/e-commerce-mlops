"""Functions to generate Datasets."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic.types import DirectoryPath

from src.core.custom_errors import MissingDataError, MissingEnvironmentVariableError


def ensure_dataset_dir_content(
    path: DirectoryPath | None,
    root_dir: DirectoryPath,
) -> str | None:
    """Ensure the dataset directory contains all the file needed.

    Args:
        path: full path to the dataset directory.
        root_dir: project root dir to define absolute path.

    Returns:
        Absolute path to the dataset directory.
    """
    logger = logging.getLogger(__name__)
    explanation = """When defined, ORIGINAL_DATA_DIR must contain:
ORIGINAL_DATA_DIR
├── X.csv  File containing the features: [linenumber,designation,description,productid,imageid]
├── images Folder containing one image per product named with this format: "image_[imageid]_product_[productid].jpg"
└── y.csv  File containing the target: prdtypecode"""  # noqa: E501

    if path is None:
        logger.warning(
            "ORIGINAL_DATA_DIR is not defined. "
            + "An error will be raised if you try to run a script needing it.\n\n"
            + explanation
        )
        return None

    full_path: str
    if Path(path).is_absolute():
        logger.debug("ORIGINAL_DATA_DIR is an absolute path.")
        full_path = str(path)
    else:
        full_path = str(Path(root_dir) / path)
        logger.debug(
            "ORIGINAL_DATA_DIR is a relative path. "
            + f"Therefore, its absolute path is {full_path}."
        )

    missing_files = get_dataset_missing_files(full_path)

    if len(missing_files) > 0:
        logger.warning(
            f"Missing file(s) in ORIGINAL_DATA_DIR set as {path}:\n- "
            + "\n- ".join(missing_files)
            + "\nAn error will be raised if you try to run code needing it(them).\n"
            + explanation
        )
        return None
    return full_path


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
    logger = logging.getLogger(__name__)
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


def to_simplified_category_id(y: np.ndarray):
    """Convert the category id into a simplified equivalent ranging from 0 to 26.

    Args:
        y: list of category id to convert to a simplified range.

    Returns:
        y with converted category id.
    """
    from src.core.settings import get_common_settings

    categories = get_common_settings().CATEGORIES_SIMPLIFIED_DIC
    return np.array([categories[i] for i in y])


def to_normal_category_id(y: np.ndarray) -> np.ndarray:
    """Convert back a simplified category id to the original category id.

    Args:
        y: list of category id to convert to a the original value.

    Returns:
        y with original category id.
    """
    from src.core.settings import get_common_settings

    categories = get_common_settings().CATEGORIES_SIMPLIFIED_DIC
    return np.array(
        [list(categories.keys())[list(categories.values()).index(i)] for i in y]
    )
