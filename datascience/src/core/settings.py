"""Contains all the settings needed in datascience project.

Keep in mind classes inheriting from BaseSettings can have their properties
replaced by environment variables. To add settings that cannot be changed,
add them in _ConstantSettings.
Note: important variable must have a validator using pydantic.
"""
import logging
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseSettings, DirectoryPath, validator

import src
from src.utilities.dataset_utils import get_dataset_missing_files


class _ConstantSettings:
    """Define constants we want embedded into the settings."""

    # TODO @JoffreyLGT: Should we have a package handling configurations since they are common to all projects?
    # https://github.com/JoffreyLGT/e-commerce-mlops/issues/80
    ROOT_DIR: DirectoryPath = Path(Path(src.__file__).parent).parent
    CATEGORIES_DIC = MappingProxyType(
        {
            10: "Livre",
            1140: "Figurine et produits dérivés",
            1160: "Carte à collectionner",
            1180: "Univers fantastiques",
            1280: "Jouet pour enfant",
            1281: "Jeu de société",
            1300: "Miniature de collection",
            1301: "Loisir",
            1302: "Activité d'extérieur",
            1320: "Accessoire bébé",
            1560: "Meuble d'intérieur",
            1920: "Litterie, rideaux",
            1940: "Epicerie",
            2060: "Décoration d'intérieur",
            2220: "Accessoire animaux de compagnie",
            2280: "Magazine et BD",
            2403: "Livres anciens",
            2462: "Jeu vidéo - Pack",
            2522: "Fourniture de bureau",
            2582: "Meubles extérieur",
            2583: "Piscine",
            2585: "Bricolage",
            2705: "Livre",
            2905: "Jeu vidéo - Jeu",
            40: "Jeu vidéo - Jeu",
            50: "Jeu vidéo - Accessoire",
            60: "Jeu vidéo - Console",
        }
    )


class _CommonSettings(BaseSettings, _ConstantSettings):
    """Common settings of the project."""

    TARGET_ENV: Literal["development", "staging", "production"] = "development"
    CONSOLE_LOG_LEVEL: Literal[
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    ] = "WARNING"
    LOGS_DIR: str = "logs"

    @validator("LOGS_DIR")
    def create_log_dir(
        cls,  # noqa: N805 false positive, first argument must be cls for validator
        path: str,
    ) -> DirectoryPath:
        """Ensure the directory contains the necessary values."""
        full_path: Path = (
            Path(path) if Path(path).is_absolute() else Path(cls.ROOT_DIR) / path
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


class _DatasetSettings(BaseSettings, _ConstantSettings):
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
    def must_contain_data(
        cls,  # noqa: N805 false positive, first argument must be cls for validator
        path: DirectoryPath | None,
    ) -> str | None:
        """Ensure the directory contains the necessary values."""
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
            logger.info("ORIGINAL_DATA_DIR is an absolute path.")
            full_path = str(path)
        else:
            full_path = str(Path(cls.ROOT_DIR) / path)
            logger.info(
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
        return full_path

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
