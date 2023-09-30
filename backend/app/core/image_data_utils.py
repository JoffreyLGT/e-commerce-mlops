"""Provides functions to manage the content of the data folder.

Data folder is a volume containing all images and used by the api and datascience
services.
"""

import uuid
from pathlib import Path

import numpy.typing as nt
from PIL import Image

from app.core.settings import settings


def generate_image_file_name(
    product_id: str | None = None, image_id: str | None = None
) -> str:
    """Return the product image name.

    Args:
        product_id: id of the product.
        image_id: id of the image.

    Returns:
        Image file name.
    """
    if product_id is None:
        product_id = str(uuid.uuid1())

    if image_id is None:
        image_id = str(uuid.uuid1())

    return f"image_{image_id}_product_{product_id}.jpg"


def save_temporary_image(image: nt.ArrayLike, product_id: str | None = None) -> Path:
    """Save image into temporary product image directory.

    Args:
        image: matrix containing the image information.
        product_id: identifier of the product.

    Return:
        Path to the image.
    """
    image_name = generate_image_file_name(product_id)
    full_path = settings.TEMPORARY_PRODUCT_IMAGES_DIR / image_name
    Image.fromarray(image).save(full_path)
    return full_path
