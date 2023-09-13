# type: ignore
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
import logging
import os
import pickle
import sys
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pydantic
import pydantic_argparse
import tensorflow as tf
from keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from keras.layers import Dense, Dropout, InputLayer
from keras.losses import CategoricalCrossentropy
from keras.optimizers import (
    Adam,
)
from pydantic import BaseModel, DirectoryPath
from sklearn import metrics
from tensorflow.train import latest_checkpoint

from src.core.settings import get_common_settings
from src.transformers.text_transformer import TextPreprocess, TfidfStemming
from src.utilities.dataset_utils import to_normal_category_id, to_simplified_category_id
from src.utilities.model_eval import (
    gen_classification_report,
    gen_confusion_matrix,
    gen_training_history_figure,
)

logger = logging.getLogger(__file__)


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
    train_patience: int = pydantic.Field(
        10,
        description=(
            "Number of epoch to do without improvements "
            "before stopping the model training."
        ),
    )


def main(args: TrainTextModelArgs) -> int:  # noqa: PLR0915
    """Create and train a new MLP text model.

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

    y_train_simplified = to_simplified_category_id(y_train)
    y_test_simplified = to_simplified_category_id(y_test)

    y_train_categorical = keras.utils.to_categorical(y_train_simplified)
    y_test_categorical = keras.utils.to_categorical(y_test_simplified)

    logger.info("Create preprocessor and preprocess features")
    preprocessor = TextPreprocess(TfidfStemming())
    X_train_tensor = preprocessor.fit_transform(X_train)
    X_test_tensor = preprocessor.transform(X_test)
    preprocessor_path = args.output_dir / "text_preprocess.pkl"
    with preprocessor_path.open("wb") as file:
        pickle.dump(preprocessor, file)
    logger.info(f"Preprocessor saved: {preprocessor_path}")

    logger.info("Create datasets")
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train_tensor, y_train_categorical)
    ).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test_tensor, y_test_categorical)
    ).batch(args.batch_size)

    # # Add cache configuration to speed up training
    # # If this cause issues, we'll add an argument to enable it
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

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
        train_ds,
        epochs=100,
        validation_data=test_ds,
        callbacks=cp_callbacks,
        batch_size=args.batch_size,
    )

    logger.info("Save the model")
    # Load the latest checkpoint to avoid overfitting
    latest = latest_checkpoint(checkpoints_dir)
    if latest is not None:
        model.load_weights(latest)
    model.save(args.output_dir / "mlp_text.keras")

    logger.info("Generate training history figure")
    logger.info(gen_training_history_figure(history_file_path, args.output_dir))

    logger.info("Predict test data categories")
    y_pred_simplified = model.predict(test_ds)
    y_pred = to_normal_category_id([np.argmax(i) for i in y_pred_simplified])

    logger.info(f"Accuracy score: {metrics.accuracy_score(y_test, y_pred)}")

    logger.info("Generate classification report")
    (_, class_report) = gen_classification_report(y_test, y_pred, args.output_dir)
    logger.info(class_report)

    logger.info("Generate confusion matrix")
    logger.info(gen_confusion_matrix(y_test, y_pred, args.output_dir))

    logger.info("Script finished")
    return 0


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=TrainTextModelArgs,
        description=(
            "Create and train a new text model "
            "using dataset provided with --input-dir, "
            "then save it to --output-dir with "
            "its performance statistics."
        ),
    )
    args = parser.parse_typed_args()
    sys.exit(main(args))
