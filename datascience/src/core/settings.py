"""Contains all the settings needed to run the API.

Most of them are coming from env variables, others are
constructed or simply defined.
Note: important variable must have validator using pydantic.
"""
import logging
import os
from typing import Literal

import datascience
from pydantic import (
    BaseSettings,
    DirectoryPath,
    validator,
)

datascience_dir: str = os.path.dirname(datascience.__file__)


class CommonSettings(BaseSettings):
    """Common settings of the project."""

    TARGET_ENV: Literal["development", "staging", "production"] = "development"
    CONSOLE_LOG_LEVEL: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "WARNING"


class DatasetSettings(CommonSettings):
    """Settings to manage datasets."""

    # Preprocessed target images characteristics
    IMG_WIDTH: int = 224
    IMG_HEIGHT: int = 224
    IMG_KEEP_RATIO: bool = True
    IMG_GRAYSCALED: bool = False
    # Number of threads to use when doing images conversion
    IMG_NB_THREADS: int = 4
    # Directory where data used to train the models are stored.
    ORIGINAL_DATA_DIR: DirectoryPath | None = None

    @validator("ORIGINAL_DATA_DIR")
    def must_contain_data(cls, path: str | None):
        """Ensure the directory contains the necessary values."""
        logger = logging.getLogger(__name__)

        explanation = """When defined, ORIGINAL_DATA_DIR must contain:
ORIGINAL_DATA_DIR
├── X.csv  File containing the features: [linenumber,designation,description,productid,imageid]
├── images Folder containing one image per product named with this format: "image_[imageid]_product_[productid].jpg"
└── y.csv  File containing the target: prdtypecode
        """  # noqa: E501

        if path is None:
            logger.warning(
                "ORIGINAL_DATA_DIR is not defined. "
                + "An error will be raised if you try to run a script needing it.\n\n"
                + explanation
            )
            return None

        full_path: str

        if os.path.isabs(path):
            logger.info("ORIGINAL_DATA_DIR is an absolute path.")
            full_path = path
        else:
            full_path = os.path.join(datascience_dir, path)
            logger.info(
                "ORIGINAL_DATA_DIR is a relative path. "
                + f"Therefore, its absolute path is {full_path}."
            )

        missing_files: list[str] = []
        if not os.path.isfile(f"{full_path}/X.csv"):
            missing_files.append("X.csv")
        if not os.path.isfile(f"{full_path}/y.csv"):
            missing_files.append("y.csv")
        if not os.path.isdir(f"{full_path}/images"):
            missing_files.append("images")

        if len(missing_files) > 0:
            logger.warning(
                f"Missing files in ORIGINAL_DATA_DIR set as {path}:\n- "
                + "\n- ".join(missing_files)
                + "An error will be raised if you try to run a script needing it.\n"
                + explanation
            )

    # Directory where remaining data not already in a dataset are stored.
    REMAINING_DATA_DIR: str | None = None
