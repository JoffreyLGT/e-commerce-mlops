"""Functions to generate Datasets."""
import itertools
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic.types import DirectoryPath

from src.core import constants
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


def to_simplified_category_id(
    y: np.ndarray[Any, np.dtype[np.int32]]
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """Convert the category id into a simplified equivalent ranging from 0 to 26.

    Args:
        y: list of category id to convert to a simplified range.

    Returns:
        y with converted category id.
    """
    categories = constants.CATEGORIES_SIMPLIFIED_DIC
    return np.array([categories[i] for i in y])


def to_normal_category_id(
    y: Iterable[int],  # np.ndarray[Any, np.dtype[np.int32]]
) -> Iterable[int]:  # np.ndarray[Any, np.dtype[np.int32]]:
    """Convert back a simplified category id to the original category id.

    Args:
        y: list of category id to convert to a the original value.

    Returns:
        y with original category id.
    """
    categories = constants.CATEGORIES_SIMPLIFIED_DIC
    return np.array(
        [list(categories.keys())[list(categories.values()).index(i)] for i in y]
    )


def get_img_name(productid: int, imageid: int) -> str:
    """Return the filename of the image.

    Args:
        productid: "productid" field from the original DataFrame.
        imageid: "imageid" field from the original DataFrame.

    Returns:
        Image filename, for example: image_1000076039_product_580161.jpg
    """
    return f"image_{imageid}_product_{productid}.jpg"


def get_imgs_filenames(
    productids: list[int], imageids: list[int], img_dir: Path
) -> list[str]:
    """Return a list of filenames from productids and imagesids.

    Args:
        productids: list of product ids
        imageids: list of image ids
        img_dir: dir containing the images. Used only to return a full path.

    Returns:
        A list of the same size as productids and imageids containing the filenames.
    """
    if len(productids) != len(imageids):
        raise ValueError(  # noqa: TRY003
            "productids and imageids should be the same size"
        )
    if img_dir is None:
        return [
            get_img_name(productid, imageid)
            for productid, imageid in zip(productids, imageids)
        ]
    return [
        str(img_dir / get_img_name(productid, imageid))
        for productid, imageid in zip(productids, imageids)
    ]


def to_img_feature_target(filename: str, y: Any = None) -> tuple[tf.Tensor, Any]:
    """Open image and return a resized version in a tensor with the target.

    Args:
        filename: complete path to image file including the extension.
        y: image category.

    Return:
        Tuple with (image matrix in a tensor, category to predict).
    """
    img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img, channels=3)
    return (tf.image.resize(img, [224, 224]), y)


def convert_sparse_matrix_to_sparse_tensor(X) -> tf.SparseTensor:  # type: ignore
    """Convert provided sparse matrix into a sparse tensor."""
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))


class CategoryProbabilities(TypedDict):
    """Represent the probabilities for an item to be part of a category."""

    category_id: int
    probabilities: float


def get_probabilities_header() -> list[str]:
    """Generates the header line for a category probabilities array."""
    return list(
        itertools.chain(
            ["product_id"], [str(i) for i in constants.CATEGORIES_SIMPLIFIED_DIC]
        )
    )


def get_empty_product_category(
    product_ids: Iterable[str], with_header: bool = True
) -> list[list[str]]:
    """Return a list with product_id has the first item rest with empty strings.

    Args:
        product_ids: to fill as first item into the list.
        with_header: true for the first line to describe the columns.

    Returns:
        List with a header and one line per product id.
    """
    list_decisions: list[list[str]] = list()

    if with_header:
        list_decisions.append(get_probabilities_header())

    nb_categories = len(constants.CATEGORIES_SIMPLIFIED_DIC.keys())

    empty_values = [
        list(itertools.chain([product_id], ["" for _ in range(0, nb_categories)]))
        for product_id in product_ids
    ]
    return list(itertools.chain(list_decisions, empty_values))


def get_product_category_probabilities(
    product_ids: Iterable[str],
    y_pred_simplified: list[list[float]],
    with_header: bool = True,
) -> list[Sequence[str | int | float]]:
    """Get the probabilities for each categories from the predictions.

    Args:
        product_ids: list of product ids.
        y_pred_simplified: predictions from the model.
        with_header: true for the first line to describe the columns.

    Returns:
        A list with the product id and prediction probabilities for each category.
    """
    list_decisions: list[Sequence[str | int | float]] = list()
    if with_header:
        probabilities = get_probabilities_header()
        list_decisions.append(probabilities)

    for product_id, y in zip(product_ids, y_pred_simplified):
        list_probabilities: list[str | int | float] = list()
        list_probabilities.append(product_id)
        nb_category = len(constants.CATEGORIES_SIMPLIFIED_DIC.keys())
        for i, probability in enumerate(y):
            if nb_category <= i:
                raise NotImplementedError(
                    f"There are only {nb_category} categories, not {i}."
                )
            list_probabilities.append(np.around(probability * 100, 2))
        list_decisions.append(list_probabilities)
    return list_decisions
