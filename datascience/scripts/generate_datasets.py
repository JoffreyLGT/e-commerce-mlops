# type: ignore
"""Optimize images provided by Rakuten to improve image model initial training.

Open provided folder, create train and test dataset and optimize images by:
- Removing the white stripes they can have around
- Resize
- Keep the ratio (by adding white stripes if needed) or stretch images
- Save it either grayscaled or colored
"""

import datetime
import logging
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
import pydantic_argparse
from PIL import Image, ImageOps
from pydantic import BaseModel, DirectoryPath, Field, validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

from src.core.custom_errors import ImageProcessingError
from src.core.settings import (
    get_common_settings,
    get_dataset_settings,
    get_mobilenet_image_model_settings,
)
from src.utilities.dataset_utils import ensure_dataset_dir_content

logger = logging.getLogger(__file__)


def get_img_name(productid: int, imageid: int) -> str:
    """Return the filename of the image.

    Args:
        productid: "productid" field from the original DataFrame.
        imageid: "imageid" field from the original DataFrame.

    Returns:
        A string containing the filename of the image. Example: image_1000076039_product_580161.jpg
    """  # noqa: E501
    return f"image_{imageid}_product_{productid}.jpg"


def get_imgs_filenames(
    productids: list[int], imageids: list[int], folder: str | None = None
) -> list[str]:
    """Return a list of filenames from productids and imagesids.

    Args:
        productids: list of product ids
        imageids: list of image ids
        folder: folder containing the images. Used only to return a full path.

    Returns:
        A list of the same size as productids and imageids containing the filenames.
    """
    if len(productids) != len(imageids):
        raise ValueError(  # noqa: TRY003
            "productids and imageids should be the same size"
        )
    if folder is None:
        return [
            get_img_name(productid, imageid)
            for productid, imageid in zip(productids, imageids)
        ]
    return [
        Path(folder) / get_img_name(productid, imageid)
        for productid, imageid in zip(productids, imageids)
    ]


def remove_white_stripes(img_array: np.ndarray) -> np.ndarray:
    """Remove image outer white stripes.

    Args:
        img_array: image loaded into a np.ndarray.

    Returns:
        The same array without the outer white stripes.

    Example:
        remove_white_stripes(np.asarray(Image.open("my_image.png")))
    """
    top_line = -1
    right_line = -1
    bottom_line = -1
    left_line = -1

    i = 1
    white_color_average = 255

    while top_line == -1 or bottom_line == -1 or left_line == -1 or right_line == -1:
        if top_line == -1 and img_array[:i].mean() != white_color_average:
            top_line = i
        if bottom_line == -1 and img_array[-i:].mean() != white_color_average:
            bottom_line = i
        if left_line == -1 and img_array[:, :i].mean() != white_color_average:
            left_line = i
        if right_line == -1 and img_array[:, -i:].mean() != white_color_average:
            right_line = i

        i += 1
        if i >= img_array.shape[0]:
            break

    if top_line == -1 or bottom_line == -1 or left_line == -1 or right_line == -1:
        return img_array
    return img_array[top_line:-bottom_line, left_line:-right_line, :]


def crop_resize_img(  # noqa: PLR0913
    filename: str,
    input_img_dir: str,
    output_img_dir: str,
    width: int,
    height: int,
    keep_ratio: bool,
    grayscale: bool = False,
) -> None:
    """Crop, resize and apply a grayscale filter to the image.

    Args:
        imput_img_dir: dir containing all images.
        filename: name of the image to process. Must contain the extension.
        input_img_dir: directory containing the image.
        output_img_dir: directory to save the processed image in.
        width: width of the processed image
        height: height of the processed image.
        keep_ratio: True to keep the image ratio and eventualy add some white stripes around to fill empty space. False to stretch the image.
        grayscale: True to remove the colors and set them as grayscale.
    """  # noqa: E501
    # Remove the outer white stripes from the image
    img_array = np.asarray(Image.open(Path(input_img_dir) / filename))
    new_img_array = remove_white_stripes(img_array)
    new_img = Image.fromarray(new_img_array)

    if keep_ratio:
        new_width = new_img.width
        new_height = new_img.height

        ratio = new_width - new_height
        padding_value = np.abs(ratio) // 2
        padding = ()
        if ratio > 0:
            padding = (0, padding_value, 0, padding_value)
        else:
            padding = (padding_value, 0, padding_value, 0)

        new_img = ImageOps.expand(new_img, padding, (255, 255, 255))

    new_img = new_img.resize((width, height))

    if grayscale:
        new_img = ImageOps.grayscale(new_img)

    new_img.save(f"{output_img_dir}/{filename}")


class Progression:
    """Inform the user about the progression of images transformation."""

    def __init__(self, total_rows: int):
        """Initiate a Progression object."""
        self.start_time = time.perf_counter()
        self.total_rows = total_rows
        self.nb_calls = 0

    def display(self, remaining_rows_number: int):
        """Display the image progression in the output."""
        current_row_number = self.total_rows - remaining_rows_number
        if current_row_number == 0:
            return
        time_diff = time.perf_counter() - self.start_time
        time_per_row = time_diff / current_row_number
        remaining_time = (self.total_rows - current_row_number) * time_per_row
        if self.nb_calls == 0:
            # To print an empty line
            print("")
        else:
            self.nb_calls += 1
        # Replace previously printed line
        sys.stdout.write(
            "\033[F"  # Go up by one line in the console
            f"Progress: {np.round(current_row_number / self.total_rows * 100, 2)}"
            f"%, Time remaining: {datetime.timedelta(seconds=int(remaining_time))}"
            "\033[K"  # Clear the rest of the line
        )
        sys.stdout.flush()

    def done(self):
        """Display completion time."""
        print()  # Print an empty line since display was always writing on the same one
        logger.info(
            "Completed in "
            f"{np.round(time.perf_counter() - self.start_time, 2)} seconds."
        )
        self.nb_calls = 0


class ImageTransformer(BaseEstimator, TransformerMixin):
    """Transform images.

    First by cropping the white stripes and then resize them either
    while keeping their original ratio or by stretching them.
    Can also set it to grayscale.
    """

    def __init__(  # noqa: PLR0913
        self,
        size: tuple[int, int],
        keep_ratio: bool,
        grayscale: bool,
        input_img_dir: str,
        output_img_dir: str,
    ) -> None:
        """Store arguments into object properties.

        Args:
            size: image width and height
            keep_ratio: true to keep image ratio (add white stripes if needed).
            grayscale: true to set image as grayscale.
            input_img_dir: original images directory.
            output_img_dir: directory to store edited images.
        """
        super().__init__()
        self.width = size[0]
        self.height = size[1]
        self.keep_ratio = keep_ratio
        self.grayscale = grayscale
        self.input_img_dir = input_img_dir
        self.nb_threads = get_dataset_settings().IMG_PROCESSING_NB_THREAD
        self.output_img_dir = output_img_dir
        self.filenames_queue = Queue()

    def _initiate_crop_resize(self, type):
        """Private function started by the threads to crop_resize_img."""
        while not self.filenames_queue.empty():
            filename, prdtypecode = self.filenames_queue.get()
            output_dir = (
                self.output_img_dir
                if prdtypecode is None
                else Path(self.output_img_dir) / type / str(prdtypecode)
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            crop_resize_img(
                filename,
                self.input_img_dir,
                output_dir,
                self.width,
                self.height,
                self.keep_ratio,
                self.grayscale,
            )

    def _load_images_to_dataframe(self, features, filenames: list[str]):
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
        features = pd.concat(
            [features.reset_index(drop=True), df_pixels.reset_index(drop=True)], axis=1
        )
        return features.drop(
            ["designation", "description", "imageid", "productid"], axis=1
        )

    def transform(self, x, y=None, type: str | None = None):
        """Transform the images for each line of X.

        Args:
            x: features
            y: targets. When provided, images will be placed in a subfolder. Defaults to None.
            type: dataset type. For example: test, train, validation... Defaults to None.

        Raises:
            ImageProcessingError: when an error occurs during the processing.
        """  # noqa: E501
        # Create the list of images to import from X
        images_filenames = get_imgs_filenames(x["productid"], x["imageid"])

        if y is None:
            files_to_process = list(
                filter(
                    lambda value: value is not None,
                    [(x, None) for x in images_filenames],
                )
            )
        else:
            files_to_process = list(
                filter(
                    lambda value: value is not None,
                    [(x, y) for x, y in zip(images_filenames, y)],
                )
            )

        # Update the queue with the list of images to process
        self.filenames_queue.queue = deque(files_to_process)

        # Create the threads and start them
        threads = [
            Thread(target=self._initiate_crop_resize, args={type: type})
            for _ in range(self.nb_threads)
        ]
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
            time.sleep(0.1)

        # Security: wait until all threads are done
        for thread in threads:
            thread.join()

        if self.filenames_queue.qsize() != 0:
            raise ImageProcessingError

        # Inform the user
        progress.done()


class OptimizeImagesArgs(BaseModel):
    """Hold all scripts arguments and do type checking."""

    train_size: int | float = Field(
        description=(
            "Size of train dataset. Provide a float for a percentage, "
            "an int for the number of products."
        )
    )
    test_size: int | float = Field(
        description=(
            "Size of test dataset. Provide a float for a percentage, "
            "an int for the number of products."
        )
    )
    input_dir: DirectoryPath = Field(
        descriptions=(
            "Directory containing data to divide into train and test datasets."
        )
    )
    output_dir: Path = Field(descriptions=("Directory to store the results"))

    @validator("input_dir")
    def must_contain_data(
        cls,  # noqa: N805 false positive, first argument must be cls for validator
        path: DirectoryPath | None,
    ) -> str | None:
        """Ensure the directory contains the necessary values."""
        return ensure_dataset_dir_content(
            path=path, root_dir=Path(get_common_settings().ROOT_DIR)
        )


def main(args: OptimizeImagesArgs) -> int:
    """Create a dataset of images and optimize them.

    Args:
        args: settings provided as arguments.
    """
    logger.info("\n🚀 Script started\n")
    logger.info(
        f"Remove csv files and images folder from {args.output_dir} "
        "to ensure it contains only newly generated data."
    )
    shutil.rmtree(Path(args.output_dir) / "images", ignore_errors=True)
    for csv in Path.glob("*.csv"):
        Path.remove(csv)
    (Path(args.output_dir) / "images").mkdir(parents=True)

    model_settings = get_mobilenet_image_model_settings()

    logger.info("Import input data")
    features = pd.read_csv(f"{args.input_dir}/X.csv", index_col=0)
    target = pd.read_csv(f"{args.input_dir}/y.csv", index_col=0)

    logger.info("Create train and test datasets")
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        train_size=args.train_size,
        test_size=args.test_size,
        stratify=target,
    )

    logger.info("Save datasets")
    X_train.to_csv(f"{args.output_dir}/X_train.csv")
    X_test.to_csv(f"{args.output_dir}/X_test.csv")
    y_train.to_csv(f"{args.output_dir}/y_train.csv")  # pyright: ignore
    y_test.to_csv(f"{args.output_dir}/y_test.csv")  # pyright: ignore

    pipeline = ImageTransformer(
        size=(model_settings.IMG_WIDTH, model_settings.IMG_HEIGHT),
        keep_ratio=model_settings.IMG_KEEP_RATIO,
        grayscale=model_settings.IMG_GRAYSCALED,
        input_img_dir=str(Path(args.input_dir) / "images"),
        output_img_dir=str(Path(args.output_dir) / "images"),
    )

    logger.info("Transform images from training dataset")
    pipeline.transform(X_train, type="train")
    logger.info("Transform images from test dataset")
    pipeline.transform(X_test, type="test")
    logger.info("\n✅ done!\n")
    return 0


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=OptimizeImagesArgs,
        description=(
            "Create train and test datasets based on --input-dir and "
            "store it in --output-dir."
        ),
    )
    args = parser.parse_typed_args()
    sys.exit(main(args))
