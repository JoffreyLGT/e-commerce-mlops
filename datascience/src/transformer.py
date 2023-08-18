import datetime
import os
import queue
import re
import time
from html.parser import HTMLParser
from queue import Queue
from threading import Thread

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from datascience.src.data import crop_resize_img, get_imgs_filenames


class HTMLRemover(BaseEstimator, TransformerMixin):
    """Transformer removing HTML tags and decoding HTML special characters."""

    def _parseValue(self, value):
        if type(value) != str:
            return value
        parser = _RakutenHTMLParser()
        parser.feed(value)
        return parser.get_all_content()

    def _parseColumn(self, column):
        return [self._parseValue(value) for value in column]

    def fit(self, X, y=None):
        # Do nothing, mandatory function for when a model is provided to the pipeline.
        return self

    def transform(self, X):
        if type(X) == pd.DataFrame:
            return X.apply(lambda column: self._parseColumn(column))

        return X.apply(lambda column: self._parseValue(column))


class NumRemover(BaseEstimator, TransformerMixin):
    def _parseValue(self, value):
        if type(value) != str:
            return value
        value = re.sub('\s?([0-9]+)\s?', ' ', value)
        return value

    def _parseColumn(self, column):
        return [self._parseValue(value) for value in column]

    def fit(self, X, y=None):
        # Do nothing, mandatory function for when a model is provided to the pipeline.
        return self

    def transform(self, X):
        if type(X) == pd.DataFrame:
            return X.apply(lambda column: self._parseColumn(column))

        return X.apply(lambda column: self._parseValue(column))


class _RakutenHTMLParser(HTMLParser):
    """Parse the text fed to it using feed() and return the content without HTML tag or encoding with get_all_content()."""

    def __init__(self):
        self.allcontent = ""
        super().__init__()

    def handle_data(self, data):
        self.allcontent += data + " "

    def get_all_content(self):
        return self.allcontent.strip()


class Progression:
    """Inform the user about the progression of images transformation."""

    def __init__(self, total_rows: int):
        self.start_time = time.perf_counter()
        self.total_rows = total_rows

    def display(self, remaining_rows_number: int):
        """Display the image progression in the output."""
        current_row_number = self.total_rows - remaining_rows_number
        if current_row_number == 0:
            return
        time_diff = time.perf_counter() - self.start_time
        time_per_row = time_diff / current_row_number
        remaining_time = (self.total_rows - current_row_number) * time_per_row

        print("Avancement : ", np.round(
            current_row_number/self.total_rows*100, 2), "%")
        print("Temps restant :", datetime.timedelta(
            seconds=int(remaining_time)))

    def done(self):
        """Clear the output and display a 100% progress."""
        print("Avancement : 100%")


class ImageTransformer(BaseEstimator, TransformerMixin):
    """Transform images by first cropping the white stripes and then resize them either while keeping their original ratio or by stretching them. Can also set it to grayscale."""

    def __init__(self, width: int, height: int, keep_ratio: bool, grayscale: bool, input_img_dir: str = "data/images/image_train/", nb_threads: int = 4) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.keep_ratio = keep_ratio
        self.grayscale = grayscale
        self.input_img_dir = input_img_dir
        self.nb_threads = nb_threads
        self.output_img_dir = self.get_output_dir()
        self.filenames_queue = Queue()

    def _initiate_crop_resize(self, type):
        """Private function started by the threads to crop_resize_img."""
        while not self.filenames_queue.empty():
            filename, prdtypecode = self.filenames_queue.get()
            output_dir = self.output_img_dir if prdtypecode is None else os.path.join(
                self.output_img_dir,
                type,
                str(prdtypecode))
            os.makedirs(output_dir, exist_ok=True)
            crop_resize_img(filename, self.input_img_dir, output_dir,
                            self.width, self.height, self.keep_ratio, self.grayscale)

    def _load_images_to_dataframe(self, X,  filenames: list[str]):
        images = np.array([np.asarray(Image.open(
            f"{self.output_img_dir}/{filename}")) for filename in filenames])
        if self.grayscale:
            df_pixels = pd.DataFrame(images.flatten().reshape(
                images.shape[0], images.shape[1]*images.shape[2]))
        else:
            df_pixels = pd.DataFrame(images.flatten().reshape(
                images.shape[0], images.shape[1]*images.shape[2]*images.shape[3]))
        X = pd.concat([X.reset_index(drop=True),
                      df_pixels.reset_index(drop=True)], axis=1)
        X = X.drop(["designation", "description",
                    "imageid", "productid"], axis=1)

        return X

    def get_output_dir(self):
        """Return the output directory where transformed images will be saved."""
        result = f"data/images/cropped_w{self.width}_h{self.height}"
        if self.keep_ratio:
            result += "_ratio"
        else:
            result += "_stretched"
        if self.grayscale:
            result += "_graycaled"
        else:
            result += "_colors"
        return result

    def fit(self, X=None, y=None):
        """Doesn't do anything in this scenario, but must be here for Pipeline compatibility."""
        return self

    def transform(self, X, y=None, type: str | None = None):
        """Transform the images for each line of X."""
        existing_files = []

        # Check if the output directory for images exists
        if os.path.exists(self.output_img_dir):
            # It does, check its content
            existing_files = []
            for path, subdirs, files in os.walk(self.output_img_dir):
                for name in files:
                    existing_files.append(name)

        # Create the list of images to import from X
        images_filenames = get_imgs_filenames(
            X["productid"], X["imageid"])

        # Remove the images already in the destination folder so we don't have to process them a second time
        if y is None:
            files_to_process = list(filter(lambda value: value is not None, [(
                (x, None) if x not in existing_files else None) for x in images_filenames]))
        else:
            files_to_process = list(filter(lambda value: value is not None, [(
                (x, y) if x not in existing_files else None) for x, y in zip(images_filenames, y)]))

        # Update the queue with the list of images to process
        self.filenames_queue.queue = queue.deque(files_to_process)

        # Create the threads and start them
        threads = []
        for i in range(self.nb_threads):
            threads.append(
                Thread(target=self._initiate_crop_resize, args={type: type}))
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
            raise Exception(
                "All threads were are stopped but there are still some images in the list.")

        # Inform the user
        progress.done()

        if y is None:
            return self._load_images_to_dataframe(X, images_filenames)
        else:
            return X


class ImagePipeline:

    def __init__(self, width: int, height: int, keep_ratio: bool, grayscale: bool, input_img_dir: str = "data/images/image_train/", nb_threads: int = 4) -> None:

        self.img_transformer = ImageTransformer(width, height,
                                                keep_ratio, grayscale, input_img_dir, nb_threads)

        self.output_dir = self.img_transformer.output_img_dir

        self.pipeline = Pipeline(steps=[
            ("ImageTransformer", self.img_transformer),
            # ("MinMaxScaler", MinMaxScaler()),
        ])


class AdvancedImagePipeline:
    """Version of ImagePipeline made for deep learning models."""

    def __init__(self, width: int, height: int, keep_ratio: bool, grayscale: bool, input_img_dir: str = "data/images/image_train/", nb_threads: int = 4) -> None:

        img_transformer = ImageTransformer(width, height,
                                           keep_ratio, grayscale, input_img_dir, nb_threads)

        self.output_dir = img_transformer.output_img_dir

        self.pipeline = Pipeline(steps=[
            ("ImageTransformer", img_transformer),
            # ("MinMaxScaler", MinMaxScaler()),
        ])
