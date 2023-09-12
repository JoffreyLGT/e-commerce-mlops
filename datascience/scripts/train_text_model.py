# ruff: noqa

"""Train text model, evaluate its performance and generates figures and stats.

All logs and best checkpoints are stored in --output-dir.
--input-dir expects a dataset with this structure is expected:

dataset_dir
├── X_test.csv
├── X_train.csv
├── images
│   ├── test
│   │   └── 2705 # One folder per category id
│   │       └── image_977803476_product_278535420.jpg
│   └── train
│       └── 2583 # One folder per category id
│           └── image_1174586892_product_2940638801.jpg
├── y_test.csv
└── y_train.csv

Use scripts/optimize_images.py to generate a dataset.
"""
from html.parser import HTMLParser
from src.core.settings import get_common_settings
import numpy.typing as npt
from src.utilities.dataset_utils import to_normal_category_id, to_simplified_category_id
import itertools
import logging
import os
import pprint
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time
from sklearn.model_selection import learning_curve, train_test_split
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.layers import Dropout

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.optimizers import Adam
import scipy.sparse as sparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import pydantic_argparse
import tensorflow as tf
from nltk.stem.snowball import SnowballStemmer
from pydantic import BaseModel, DirectoryPath
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from tensorflow.keras import Sequential, layers  # pyright: ignore
from tensorflow.keras.callbacks import (  # pyright: ignore
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.losses import CategoricalCrossentropy  # pyright: ignore
from tensorflow.keras.optimizers import SGD  # pyright: ignore
from tensorflow.train import latest_checkpoint  # pyright: ignore
import src.core.logging_config  # noqa # pyright: ignore

logger = logging.getLogger(__file__)


class TextPreprocess:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, data):
        return self.pipeline.fit(data)

    def fit_transform(self, data):
        out = self.pipeline.fit_transform(data)
        return convert_sparse_matrix_to_sparse_tensor(out)

    def transform(self, data):
        out = self.pipeline.transform(data)
        return convert_sparse_matrix_to_sparse_tensor(out)

    def get_voc(self):
        voc = self.pipeline.get_voc()
        logger.info(f"TextPreprocess.save_voc size {len(voc)}")
        return voc

    def save_voc(self, prefix_filename):
        voc = self.get_voc()
        file_name = f"{prefix_filename}_{self.pipeline.name}.pkl"
        with open(file_name, "wb") as fp:
            pickle.dump(voc, fp)
        logger.info(f"TextPreprocess.save_voc {file_name}")
        return file_name


class _RakutenHTMLParser(HTMLParser):
    """
    Parse the text fed to it using feed() and return the content without HTML tag or encoding with get_all_content().
    """

    def __init__(self):
        self.allcontent = ""
        super(_RakutenHTMLParser, self).__init__()

    def handle_data(self, data):
        self.allcontent += data + " "

    def get_all_content(self):
        return self.allcontent.strip()


class HTMLRemover(BaseEstimator, TransformerMixin):
    """
    Transformer removing HTML tags and decoding HTML special characters.
    """

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
        value = re.sub("\s?([0-9]+)\s?", " ", value)
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


class StemmedCountVectorizer(CountVectorizer):
    fr_stemmer = SnowballStemmer("french")

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (
            StemmedCountVectorizer.fr_stemmer.stem(w) for w in analyzer(doc)
        )


class StemmedTfidfVectorizer(TfidfVectorizer):
    fr_stemmer = SnowballStemmer("french")

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (
            StemmedTfidfVectorizer.fr_stemmer.stem(w) for w in analyzer(doc)
        )


class TfidfStemming(Pipeline):
    def __init__(self):
        self.name = "TfidfStemming"
        steps = [
            ("remove_html", HTMLRemover()),
            ("remove_num", NumRemover()),
            ("tfidStem", StemmedTfidfVectorizer()),
        ]
        Pipeline.__init__(self, steps)

    def get_voc(self):
        return self.steps[2][1].vocabulary_


class BOW_Stemming(Pipeline):
    """Bag of Words with french stemming"""

    def __init__(self):
        self.name = "BOW_Stemming"
        steps = [
            ("remove_html", HTMLRemover()),
            ("remove_num", NumRemover()),
            ("count", StemmedCountVectorizer()),
        ]
        Pipeline.__init__(self, steps)

    def get_voc(self):
        countvect_pos = [k for k, _ in self.steps].index("count")
        return self.steps[countvect_pos][1].vocabulary_


class TrainTextModelArgs(BaseModel):
    """Hold all scripts arguments and do type checking."""

    input_dir: DirectoryPath = pydantic.Field(
        description="Directory containing the datasets to use."
    )
    output_dir: Path = pydantic.Field(
        description="Directory to save trained model and stats."
    )
    batch_size: int = pydantic.Field(
        96,
        description=(
            "Size of the batches to use for the training. "
            "Set as much as your machine allows."
        ),
    )
    seed: int = pydantic.Field(
        123,
        description=(
            "Seed to use for randomisation. "
            "Ensure batches are always de same "
            "when executing script with the same seed."
        ),
    )
    train_patience: int = pydantic.Field(
        10,
        description=(
            "Number of epoch to do without improvements "
            "before stopping the model training."
        ),
    )


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))


def main(args: TrainTextModelArgs) -> int:  # noqa: PLR0915
    """Create and train a new MobileNetV2 based image model.

    All artifacts are exported into output-dir.

    Args:
        args: arguments provided when calling the script.

    Return:
        0 if everything went as expected, 1 otherwise.
    """
    logger.info("Script started")

    # Ensure all messages from Tensorflow will be logged and not only printed
    tf.keras.utils.disable_interactive_logging()
    # Avoid Tensorflow flooding the console with messages
    # Especially annoying with tensorflow-metal
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    logger.info("Load data")

    df_X_train = pd.read_csv(args.input_dir / "X_train.csv", index_col=0).fillna("")
    df_X_test = pd.read_csv(args.input_dir / "X_test.csv", index_col=0).fillna("")

    logger.info("Extract features and target")
    X_train = df_X_train["designation"] + " " + df_X_train["description"]
    X_test = df_X_test["designation"] + " " + df_X_test["description"]

    y_train = np.loadtxt(
        args.input_dir / "y_train.csv",
        dtype=int,
        delimiter=",",
        skiprows=1,
        usecols=(1),
    )
    y_test = np.loadtxt(
        args.input_dir / "y_test.csv", dtype=int, delimiter=",", skiprows=1, usecols=(1)
    )

    logger.info("Create preprocessor and preprocess features")
    preprocessor = TextPreprocess(TfidfStemming())
    X_train_tensor = preprocessor.fit_transform(X_train)
    X_test_tensor = preprocessor.transform(X_test)

    y_train_simplified = to_simplified_category_id(y_train)
    y_test_simplified = to_simplified_category_id(y_test)

    y_train_categorical = tf.keras.utils.to_categorical(y_train_simplified)
    y_test_categorical = tf.keras.utils.to_categorical(y_test_simplified)

    logger.info("Create datasets")
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train_tensor, y_train_categorical)
    ).batch(args.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test_tensor, y_test_categorical)
    ).batch(args.batch_size)

    # # Add cache configuration to speed up training
    # # If this cause issues, we'll add an argument to enable it
    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=autotune)
    test_dataset = test_dataset.cache().prefetch(buffer_size=autotune)

    logger.info("Build text model")
    model = tf.keras.models.Sequential()
    model.add(InputLayer(input_shape=(X_train_tensor.shape[1]), sparse=True))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(rate=0.2))

    settings = get_common_settings()
    nb_output_classes = len(settings.CATEGORIES_DIC.keys())

    model.add(Dense(units=nb_output_classes, activation="softmax"))

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoint_file_path = (
        checkpoints_dir / "cp_loss-{val_loss:.2f}_acc-{val_accuracy:.2f}.ckpt"
    )
    history_file_path = args.output_dir / "history.csv"
    tensorboard_logs_dir = args.output_dir / "tensorboard_logs"

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()

    latest = latest_checkpoint(checkpoints_dir)
    if latest is not None:
        logger.info(f"Load checkpoint: {latest}")
        model.load_weights(latest)
    else:
        logger.info("No checkpoint to load")

    # Callbacks called between each epoch
    cp_callbacks = [
        # Stop the training when there is no improvement in val_accuracy for x epochs
        EarlyStopping(monitor="val_accuracy", patience=args.train_patience),
        # Save a checkpoint
        ModelCheckpoint(
            checkpoint_file_path,
            save_best_only=True,
            mode="max",
            monitor="val_accuracy",
            save_weights_only=True,
            verbose=1,
        ),
        # Insert the metrics into a CSV file
        CSVLogger(history_file_path, separator=",", append=True),
        # Log information to display them in TensorBoard
        TensorBoard(log_dir=tensorboard_logs_dir, histogram_freq=1),
    ]

    logger.info("Start model training")
    model.fit(
        train_dataset,
        epochs=100,
        validation_data=test_dataset,
        callbacks=cp_callbacks,
        batch_size=args.batch_size,
    )

    logger.info("Save the model")
    # Load the latest checkpoint to avoid overfitting
    latest = latest_checkpoint(checkpoints_dir)
    if latest is not None:
        model.load_weights(latest)
    model.save(args.output_dir / "mlp_text.keras")

    # TODO @joffreylgt: figure and stats generation should be in their own functions.
    #  https://github.com/JoffreyLGT/e-commerce-mlops/issues/103

    logger.info("Generate training history figure")
    training_history = pd.read_csv(history_file_path, delimiter=",", header=0)

    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(121)

    # TODO @joffreylgt: translate from French to English.
    #  https://github.com/JoffreyLGT/e-commerce-mlops/issues/103

    # Labels des axes
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")

    # Courbe de la précision sur l'échantillon d'entrainement
    ax1.plot(
        np.arange(1, training_history["accuracy"].count() + 1, 1),
        training_history["accuracy"],
        label="Training Accuracy",
        color="blue",
    )

    # Courbe de la précision sur l'échantillon de test
    ax1.plot(
        np.arange(1, training_history["val_accuracy"].count() + 1, 1),
        training_history["val_accuracy"],
        label="Validation Accuracy",
        color="red",
    )

    ax1.legend()
    ax1.set_title("Accuracy per epoch")

    ax2 = fig.add_subplot(122)
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epochs")

    ax2.plot(
        np.arange(1, training_history["loss"].count() + 1, 1),
        training_history["loss"],
        label="Training loss",
        linestyle="dashed",
        color="blue",
    )

    ax2.plot(
        np.arange(1, training_history["val_loss"].count() + 1, 1),
        training_history["val_loss"],
        label="Validation loss",
        linestyle="dashed",
        color="red",
    )

    ax2.legend()
    ax2.set_title("Loss per epoch")

    plt.savefig(args.output_dir / "training_history.png")

    logger.info("Predict test data categories")
    y_pred_simplified = model.predict(test_dataset)
    y_pred = to_normal_category_id([np.argmax(i) for i in y_pred_simplified])

    logger.info(f"Accuracy score: {metrics.accuracy_score(y_test, y_pred)}")
    logger.info("Generate classification report")
    class_report = metrics.classification_report(
        y_test, y_pred, zero_division=0.0  # pyright: ignore
    )

    Path(args.output_dir / "classification_report.txt").write_text(str(class_report))
    logger.info(str(class_report))

    logger.info("Generate confusion matrix")
    cnf_matrix = np.round(metrics.confusion_matrix(y_test, y_pred, normalize="true"), 2)

    classes = range(0, nb_output_classes)
    category_ids = list(settings.CATEGORIES_DIC.keys())

    plt.figure(figsize=(13, 13))

    plt.imshow(cnf_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion matrix")
    tick_marks = classes
    plt.xticks(tick_marks, category_ids)
    plt.yticks(tick_marks, category_ids)

    for i, j in itertools.product(
        range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            cnf_matrix[i, j],
            horizontalalignment="center",
            color="white" if cnf_matrix[i, j] > (cnf_matrix.max() / 2) else "black",
        )

    plt.ylabel("Real")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45)
    plt.savefig(args.output_dir / "confusion_matrix.png")

    logger.info("Script finished")
    return 0


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=TrainTextModelArgs,
        description=(
            "Create and train a new image model "
            "using dataset provided with --input-dir, "
            "then save it to --output-dir with "
            "its performance statistics."
        ),
    )
    args = parser.parse_typed_args()
    sys.exit(main(args))
