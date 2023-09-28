# ruff: noqa
"""Create training and testing datasets from input folder.

Image are preprocessed to be optimized for our image model:
- Remove the white stripes they may have around
- Resize them to match expected dimension
- Keep the ratio (by adding white stripes if needed) or stretch images
- Save it either grayscaled or colored
"""

import datetime
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np
import pandas as pd
import pydantic
import pydantic_argparse
from PIL import Image, ImageOps
from pydantic import BaseModel, DirectoryPath, validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.core import constants
from src.core.custom_errors import ImageProcessingError
from src.core.settings import (
    get_dataset_settings,
    get_mobilenet_image_model_settings,
)
from src.utilities.dataset_utils import ensure_dataset_dir_content

# TODO @joffreylgt: improvments
#  - use proper logger to log messages
#  - look to implement a real progress bar
#  - fix linter issues by removing # ruff: noqa
#  - add proper field annotation for arguments (see train_image_model)
#  https://github.com/JoffreyLGT/e-commerce-mlops/issues/103


def load_data(datadir: str = "data") -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(f"{datadir}/X.csv", index_col=0),
            pd.read_csv(f"{datadir}/y.csv", index_col=0),
        ],
        axis=1,
    )


def get_img_name(productid: int, imageid: int) -> str:
    """Return the filename of the image.

    Arguments:
    - productid: int - "productid" field from the original DataFrame.
    - imageid: int - "imageid" field from the original DataFrame.

    Return:
    A string containing the filename of the image. Example: image_1000076039_product_580161.jpg
    """
    return f"image_{imageid}_product_{productid}.jpg"


def get_imgs_filenames(
    productids: list[int], imageids: list[int], folder: str | None = None
) -> list[str]:
    """Return a list of filenames from productids and imagesids.

    Arguments:
    - productids: list of product ids
    - imageids: list of image ids
    - folder: folder containing the images. Used only to return a full path.

    Return:
    A list of the same size as productids and imageids containing the filenames.
    """
    if len(productids) != len(imageids):
        raise ValueError("productids and imageids should be the same size")
    if folder is None:
        return [
            get_img_name(productid, imageid)
            for productid, imageid in zip(productids, imageids)
        ]
    else:
        return [
            os.path.join(folder, get_img_name(productid, imageid))
            for productid, imageid in zip(productids, imageids)
        ]


def remove_white_stripes(img_array: np.ndarray) -> np.ndarray:  # type: ignore
    """Analyse each lines and column of the array to remove the outer white stripes they might contain.

    Arguments:
    - img_array: imaged loaded into a np.ndarray.

    Returns:
    - The same array without the outer white stripes.

    Example:
    - remove_white_stripes(np.asarray(Image.open("my_image.png")))
    """
    top_line = -1
    right_line = -1
    bottom_line = -1
    left_line = -1

    i = 1
    while top_line == -1 or bottom_line == -1 or left_line == -1 or right_line == -1:
        if top_line == -1 and img_array[:i].mean() != 255:
            top_line = i
        if bottom_line == -1 and img_array[-i:].mean() != 255:
            bottom_line = i
        if left_line == -1 and img_array[:, :i].mean() != 255:
            left_line = i
        if right_line == -1 and img_array[:, -i:].mean() != 255:
            right_line = i

        i += 1
        if i >= img_array.shape[0]:
            break

    if top_line == -1 or bottom_line == -1 or left_line == -1 or right_line == -1:
        return img_array
    else:
        return img_array[top_line:-bottom_line, left_line:-right_line, :]


def crop_resize_img(
    filename: str,
    imput_img_dir: str,
    output_img_dir: str,
    width: int,
    height: int,
    keep_ratio: bool,
    grayscale: bool = False,
) -> None:
    """Crop, resize and apply a grayscale filter to the image.

    Arguments:
    - filename - str: name of the image to process. Must contain the extension.
    - input_img_dir - str: directory containing the image.
    - output_img_dir - str: directory to save the processed image in.
    - width, height - int: width and height of the processed image.
    - keep_ratio - bool: True to keep the image ratio and eventualy add some white stripes around to fill empty space. False to stretch the image.
    - grayscale - bool: True to remove the colors and set them as grayscale.
    """
    # Remove the outer white stripes from the image
    img_array = np.asarray(Image.open(Path(imput_img_dir) / filename))
    new_img_array = remove_white_stripes(img_array)
    new_img = Image.fromarray(new_img_array)

    if keep_ratio:
        new_width = new_img.width
        new_height = new_img.height

        ratio = new_width - new_height
        padding_value = np.abs(ratio) // 2
        padding = ()
        if ratio > 0:
            padding = (0, padding_value, 0, padding_value)  # type:ignore
        else:
            padding = (padding_value, 0, padding_value, 0)  # type: ignore

        new_img = ImageOps.expand(new_img, padding, (255, 255, 255))  # type: ignore

    new_img = new_img.resize((width, height))

    if grayscale:
        new_img = ImageOps.grayscale(new_img)

    new_img.save(f"{output_img_dir}/{filename}")


class Progression:
    """Inform the user about the progression of images transformation."""

    def __init__(self, total_rows: int) -> None:
        """Initiate a Progression object."""
        self.start_time = time.perf_counter()
        self.total_rows = total_rows

    def display(self, remaining_rows_number: int) -> None:
        """Display the image progression in the output."""
        current_row_number = self.total_rows - remaining_rows_number
        if current_row_number == 0:
            return
        time_diff = time.perf_counter() - self.start_time
        time_per_row = time_diff / current_row_number
        remaining_time = (self.total_rows - current_row_number) * time_per_row

        print(
            "Avancement : ",
            np.round(current_row_number / self.total_rows * 100, 2),
            "%",
        )
        print("Temps restant :", datetime.timedelta(seconds=int(remaining_time)))

    def done(self) -> None:
        """Clear the output and display a 100% progress."""
        print("Avancement : 100%")


class ImageTransformer(BaseEstimator, TransformerMixin):  # type: ignore
    """Transform images.

    First by cropping the white stripes and then resize them either
    while keeping their original ratio or by stretching them.
    Can also set it to grayscale.
    """

    def __init__(
        self,
        size: tuple[int, int],
        keep_ratio: bool,
        grayscale: bool,
        input_img_dir: str,
        output_img_dir: str,
    ) -> None:
        super().__init__()
        self.width = size[0]
        self.height = size[1]
        self.keep_ratio = keep_ratio
        self.grayscale = grayscale
        self.input_img_dir = input_img_dir
        self.nb_threads = get_dataset_settings().IMG_PROCESSING_NB_THREAD
        self.output_img_dir = output_img_dir
        self.filenames_queue = Queue()  # type: ignore

    def _initiate_crop_resize(self, type: str) -> None:
        """Private function started by the threads to crop_resize_img."""
        while not self.filenames_queue.empty():
            filename, prdtypecode = self.filenames_queue.get()
            output_dir = (
                self.output_img_dir
                if prdtypecode is None
                else os.path.join(self.output_img_dir, type, str(prdtypecode))
            )
            os.makedirs(output_dir, exist_ok=True)
            crop_resize_img(
                filename,
                self.input_img_dir,
                output_dir,
                self.width,
                self.height,
                self.keep_ratio,
                self.grayscale,
            )

    def _load_images_to_dataframe(self, X: Any, filenames: list[str]) -> Any:
        images = np.array(
            [
                np.asarray(Image.open(f"{self.output_img_dir}/{filename}"))
                for filename in filenames
            ]
        )
        if self.grayscale:
            df_pixels = pd.DataFrame(
                images.flatten().reshape(
                    images.shape[0], images.shape[1] * images.shape[2]
                )
            )
        else:
            df_pixels = pd.DataFrame(
                images.flatten().reshape(
                    images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]
                )
            )
        X = pd.concat(
            [X.reset_index(drop=True), df_pixels.reset_index(drop=True)], axis=1
        )
        return X.drop(["designation", "description", "imageid", "productid"], axis=1)

    def transform(self, x: Any, y: Any = None, type: str | None = None) -> Any:
        """Transform the images for each line of X."""
        existing_files: list[str] = []

        # Check if the output directory for images exists
        if os.path.exists(self.output_img_dir):
            # It does, check its content
            existing_files = []
            for _, _, files in os.walk(self.output_img_dir):
                for name in files:
                    existing_files.append(name)

        # Create the list of images to import from X
        images_filenames = get_imgs_filenames(x["productid"], x["imageid"])

        # Remove the images already in the destination folder so we don't have to process them a second time
        if y is None:
            files_to_process = list(
                filter(
                    lambda value: value is not None,
                    [
                        ((x, None) if x not in existing_files else None)
                        for x in images_filenames
                    ],
                )
            )
        else:
            files_to_process = list(
                filter(
                    lambda value: value is not None,
                    [
                        ((x, y) if x not in existing_files else None)
                        for x, y in zip(images_filenames, y)
                    ],
                )
            )

        # Update the queue with the list of images to process
        self.filenames_queue.queue = deque(files_to_process)

        # Create the threads and start them
        threads = []
        for _ in range(self.nb_threads):
            threads.append(Thread(target=self._initiate_crop_resize, args={type: type}))
        for thread in threads:
            thread.start()

        # Instanciate a progression object
        progress = Progression(self.filenames_queue.qsize())

        # Loop until all the files where processed or all the threads are stopped
        while not self.filenames_queue.empty():
            # Check that at least one thread is running
            is_alive = False
            for thread in threads:
                if thread.is_alive():
                    is_alive = True
                    break
            if is_alive is False:
                # It's not the case, stop the loop
                break
            # Display the progression and wait for 3 secs
            progress.display(self.filenames_queue.qsize())
            time.sleep(3)

        # Security: wait until all threads are done
        for thread in threads:
            thread.join()

        if self.filenames_queue.qsize() != 0:
            raise ImageProcessingError

        # Inform the user
        progress.done()

        if y is None:
            return self._load_images_to_dataframe(x, images_filenames)
        return x


class ImagePipeline:
    """Pipeline to process images."""

    def __init__(  # noqa: PLR0913
        self,
        size: tuple[int, int],
        keep_ratio: bool,
        grayscale: bool,
        input_img_dir: str,
        output_img_dir: str,
    ) -> None:
        """Set properties."""
        self.img_transformer = ImageTransformer(
            size, keep_ratio, grayscale, input_img_dir, output_img_dir
        )

        self.pipeline = Pipeline(
            steps=[
                ("ImageTransformer", self.img_transformer),
            ]
        )


class CreateDatasetsArgs(BaseModel):
    """Hold all scripts arguments and do type checking."""

    train_size: int | float
    test_size: int | float
    input_dir: DirectoryPath
    output_dir: Path
    cat_subdir: bool = pydantic.Field(
        False,
        description=(
            "True to store the images of each "
            "category into a subdir named with the category id."
        ),
    )

    @validator("input_dir")
    def must_contain_data(
        cls,  # noqa: N805 false positive, first argument must be cls for validator
        path: DirectoryPath | None,
    ) -> str | None:
        """Ensure the directory contains the necessary values."""
        return ensure_dataset_dir_content(path=path, root_dir=Path(constants.ROOT_DIR))


def main(args: CreateDatasetsArgs) -> int:
    """Create training and testing datasets with preprocessed images.

    Args:
        args: settings provided as arguments.
    """
    # Delete and recreate the directory to ensure it will contain only generated data
    shutil.rmtree(Path(args.output_dir), ignore_errors=True)
    Path(args.output_dir).mkdir(parents=True)

    model_settings = get_mobilenet_image_model_settings()

    dataframe = load_data(str(args.input_dir))
    target = dataframe["prdtypecode"]
    features = dataframe.drop("prdtypecode", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        train_size=args.train_size,
        test_size=args.test_size,
        stratify=target,
    )

    print("Save datasets into csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(f"{args.output_dir}/X_train.csv")  # pyright: ignore
    X_test.to_csv(f"{args.output_dir}/X_test.csv")  # pyright: ignore
    y_train.to_csv(f"{args.output_dir}/y_train.csv")  # pyright: ignore
    y_test.to_csv(f"{args.output_dir}/y_test.csv")  # pyright: ignore

    transformer = ImagePipeline(
        size=(model_settings.IMG_WIDTH, model_settings.IMG_HEIGHT),
        keep_ratio=model_settings.IMG_KEEP_RATIO,
        grayscale=model_settings.IMG_GRAYSCALED,
        input_img_dir=str(Path(args.input_dir) / "images"),
        output_img_dir=str(Path(args.output_dir) / "images"),
    )

    pipeline = transformer.img_transformer

    print("Transform and store train dataset images")
    X_train = pipeline.transform(
        X_train, y_train if args.cat_subdir is True else None, type="train"
    )
    print("Transform and store test dataset images")
    X_test = pipeline.transform(
        X_test, y_test if args.cat_subdir is True else None, type="test"
    )

    return 0


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=CreateDatasetsArgs,
        description=(
            "Generate a dataset using data from --input-dir "
            "and preprocess images to remove white stripes. "
            "Datasets are saved into csv files, images in jpg."
        ),
    )
    args = parser.parse_typed_args()
    sys.exit(main(args))
