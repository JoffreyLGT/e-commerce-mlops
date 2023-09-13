# type: ignore
"""Train fusion model, evaluate its performance and generates figures and stats.

Training is done using results from both text and image models without their
classification layer.

All logs and best checkpoints are stored in --output-dir.
--input-dir expects a dataset with this structure is expected:

dataset_dir
├── X_test.csv
├── X_train.csv
├── images
│   ├── test
│   │   └── image_977803476_product_278535420.jpg
│   └── train
│       └── image_1174586892_product_2940638801.jpg
├── y_test.csv
└── y_train.csv

Use scripts/optimize_images.py to generate a dataset.
"""
import logging
import os
import pickle
import sys
from pathlib import Path, PurePath

import keras
import numpy as np
import pandas as pd
import pydantic
import pydantic_argparse
import tensorflow as tf
from keras import layers
from keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from keras.losses import CategoricalCrossentropy
from keras.optimizers import SGD
from pydantic import BaseModel, DirectoryPath, FilePath, validator
from sklearn import metrics
from tensorflow.train import latest_checkpoint

from src.core.settings import get_common_settings
from src.utilities.dataset_utils import (
    get_imgs_filenames,
    to_img_feature_target,
    to_normal_category_id,
    to_simplified_category_id,  # pyright: ignore
)
from src.utilities.model_eval import (
    gen_classification_report,
    gen_confusion_matrix,
    gen_training_history_figure,
)

logger = logging.getLogger(__file__)


def check_extension(file_path: FilePath) -> FilePath:
    """Check file extension to ensure it's .h5 or .keras.

    Args:
        file_path: path to the model file.

    Returns:
        Path to the file if the format is correct.

    Raises:
        ValueError: when the file extension is incorrect.
    """
    if PurePath(file_path).suffix in [".h5", ".keras"]:
        return file_path
    raise ValueError("Only .h5 and .keras extensions are supported")  # noqa: TRY003


class TrainFusionModelArgs(BaseModel):
    """Hold all scripts arguments and do type checking."""

    input_dir: DirectoryPath = pydantic.Field(
        description="Directory containing the datasets to use."
    )
    output_dir: Path = pydantic.Field(
        description="Directory to save trained model and stats."
    )
    text_model: FilePath = pydantic.Field(
        description="Path to the saved text model. Format must be either .h5 or .keras."
    )
    _check_text_model_extension = validator("text_model", allow_reuse=True)(
        check_extension
    )
    text_preprocessor: FilePath = pydantic.Field(
        description="Path to the text preprocessor pkl."
    )
    image_model: FilePath = pydantic.Field(
        description=(
            "Path to the saved image model. Format must be either .h5 or .keras."
        )
    )
    _check_image_model_extension = validator("image_model", allow_reuse=True)(
        check_extension
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


def main(args: TrainFusionModelArgs) -> int:  # noqa: PLR0915
    """Create and train a new fusion model.

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

    logger.info("Extract features")
    X_train_text = df_X_train["designation"] + " " + df_X_train["description"]
    X_test_text = df_X_test["designation"] + " " + df_X_test["description"]
    X_train_img = get_imgs_filenames(
        df_X_train["productid"].to_list(),
        df_X_train["imageid"].to_list(),
        args.input_dir / "images",
    )
    X_test_img = get_imgs_filenames(
        df_X_test["productid"].to_list(),
        df_X_test["imageid"].to_list(),
        args.input_dir / "images",
    )

    logger.info("Extract targets")
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

    logger.info("Load text preprocessor and preprocess text features")
    preprocessor = pickle.load(args.text_preprocessor.open("rb"))
    X_train_text_tensor = preprocessor.transform(X_train_text)
    X_test_text_tensor = preprocessor.transform(X_test_text)

    logger.info("Create text datasets")
    train_txt_ds = tf.data.Dataset.from_tensor_slices(
        (X_train_text_tensor, y_train_categorical)
    ).batch(args.batch_size)
    test_txt_ds = tf.data.Dataset.from_tensor_slices(
        (X_test_text_tensor, y_test_categorical)
    ).batch(args.batch_size)

    logger.info("Create image datasets")
    train_img_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_img, y_train_categorical))
        .map(to_img_feature_target)
        .batch(args.batch_size)
    )
    test_img_ds = (
        tf.data.Dataset.from_tensor_slices((X_test_img, y_test_categorical))
        .map(to_img_feature_target)
        .batch(args.batch_size)
    )

    # Add cache configuration to speed up training
    # If this cause issues, we'll add an argument to enable it
    autotune = tf.data.AUTOTUNE
    train_txt_ds = train_txt_ds.cache().prefetch(buffer_size=autotune)
    test_txt_ds = test_txt_ds.cache().prefetch(buffer_size=autotune)
    train_img_ds = train_img_ds.cache().prefetch(buffer_size=autotune)
    test_img_ds = test_img_ds.cache().prefetch(buffer_size=autotune)

    logger.info("Load text model without classification layers")
    text_model = tf.keras.models.load_model(args.text_model, compile=False)
    headless_text_model = tf.keras.Model(
        inputs=text_model.inputs, outputs=text_model.layers[-2].output
    )
    headless_text_model.summary()

    logger.info("Load image model without classification layers")
    image_model = tf.keras.models.load_model(args.image_model, compile=False)
    headless_image_model = tf.keras.Model(
        inputs=image_model.inputs, outputs=image_model.layers[-2].output
    )
    headless_image_model.summary()

    logger.info("Predict using text model")
    train_txt_output = headless_text_model.predict(train_txt_ds)
    test_txt_output = headless_text_model.predict(test_txt_ds)

    logger.info("Predict using image model")
    train_img_output = headless_image_model.predict(train_img_ds)
    test_img_output = headless_image_model.predict(test_img_ds)

    logger.info("Create fusion model dataset")
    train_fusion = np.concatenate((train_txt_output, train_img_output), axis=1)
    test_fusion = np.concatenate((test_txt_output, test_img_output), axis=1)

    train_fusion_ds = tf.data.Dataset.from_tensor_slices(
        (train_fusion, y_train_categorical)
    ).batch(args.batch_size)
    test_fusion_ds = tf.data.Dataset.from_tensor_slices(
        (test_fusion, y_test_categorical)
    ).batch(args.batch_size)

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoint_file_path = (
        checkpoints_dir / "cp_loss-{val_loss:.2f}_acc-{val_accuracy:.2f}.ckpt"
    )
    history_file_path = args.output_dir / "history.csv"
    tensorboard_logs_dir = args.output_dir / "tensorboard_logs"

    settings = get_common_settings()
    fusion_model = tf.keras.Sequential()
    fusion_model.add(layers.InputLayer(input_shape=(train_fusion.shape[1])))
    fusion_model.add(layers.Dense(units=512, activation="relu"))
    fusion_model.add(layers.Dropout(rate=0.2))

    fusion_model.add(layers.Dense(units=128, activation="relu"))
    fusion_model.add(layers.Dropout(rate=0.2))

    fusion_model.add(
        layers.Dense(
            len(settings.CATEGORIES_DIC.keys()), activation="softmax", name="Output"
        )
    )

    fusion_model.build((None, train_fusion.shape[1]))
    fusion_model.compile(
        optimizer=SGD(learning_rate=0.005, momentum=0.9),
        loss=CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=["accuracy"],
    )
    fusion_model.summary()

    latest = latest_checkpoint(checkpoints_dir)
    if latest is not None:
        logger.info(f"Load checkpoint: {latest}")
        fusion_model.load_weights(latest)
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
        TensorBoard(log_dir=str(tensorboard_logs_dir), histogram_freq=1),
    ]

    logger.info("Start fusion model training")
    fusion_model.fit(
        train_fusion_ds,
        epochs=100,
        validation_data=test_fusion_ds,
        callbacks=cp_callbacks,
    )

    logger.info("Save the fusion_model")
    # Load the latest checkpoint to avoid overfitting
    latest = latest_checkpoint(checkpoints_dir)
    if latest is not None:
        fusion_model.load_weights(latest)
    fusion_model.save(args.output_dir / "fusion.keras")

    logger.info("Generate training history figure")
    logger.info(gen_training_history_figure(history_file_path, args.output_dir))

    logger.info("Predict test data categories")
    y_pred_simplified = fusion_model.predict(test_fusion_ds)
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
        model=TrainFusionModelArgs,
        description=(
            "Create and train a new fusion model "
            "using dataset provided with --input-dir, "
            "then save it to --output-dir with "
            "its performance statistics."
        ),
    )
    args = parser.parse_typed_args()
    sys.exit(main(args))
