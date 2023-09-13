# type: ignore
"""Train image model, evaluate its performance and generates figures and stats.

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
import sys
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import pydantic
import pydantic_argparse
import tensorflow as tf
from keras import Sequential, layers
from keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from keras.losses import CategoricalCrossentropy
from keras.optimizers import SGD
from pydantic import BaseModel, DirectoryPath
from sklearn import metrics
from tensorflow.train import latest_checkpoint

from src.core.settings import get_mobilenet_image_model_settings
from src.utilities.dataset_utils import (
    get_imgs_filenames,
    to_img_feature_target,
    to_normal_category_id,
    to_simplified_category_id,
)
from src.utilities.model_eval import (
    gen_classification_report,
    gen_confusion_matrix,
    gen_training_history_figure,
)


class TrainImageModelArgs(BaseModel):
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
    data_augmentation: bool = pydantic.Field(
        True, description="Add layers of data augmentation to avoid early overfitting."
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


def main(args: TrainImageModelArgs) -> int:  # noqa: PLR0915
    """Create and train a new MobileNetV2 based image model.

    All artifacts are exported into output-dir.

    Args:
        args: arguments provided when calling the script.

    Return:
        0 if everything went as expected, 1 otherwise.
    """
    logger = logging.getLogger(__file__)

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
    X_train = get_imgs_filenames(
        df_X_train["productid"].to_list(),
        df_X_train["imageid"].to_list(),
        args.input_dir / "images",
    )
    X_test = get_imgs_filenames(
        df_X_test["productid"].to_list(),
        df_X_test["imageid"].to_list(),
        args.input_dir / "images",
    )

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

    logger.info("Create datasets")
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train_categorical))
        .map(to_img_feature_target)
        .batch(args.batch_size)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test_categorical))
        .map(to_img_feature_target)
        .batch(args.batch_size)
    )

    model_settings = get_mobilenet_image_model_settings()

    nb_train_images = len(X_train)
    nb_test_images = len(X_test)

    # Add cache configuration to speed up training
    # If this cause issues, we'll add an argument to enable it
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    logger.info("Build image model")
    mobilenet_layers = tf.keras.applications.MobileNetV2(
        include_top=False,  # Do not include the ImageNet classifier at the top
        weights="imagenet",  # Load weights pre-trained on ImageNet
        input_shape=(model_settings.IMG_WIDTH, model_settings.IMG_HEIGHT, 3),
    )
    # Freeze the base model
    mobilenet_layers.trainable = False

    inputs = tf.keras.Input(
        shape=(model_settings.IMG_WIDTH, model_settings.IMG_HEIGHT, 3), name="Input"
    )

    if args.data_augmentation:
        x = Sequential(
            [
                layers.RandomFlip("horizontal_and_vertical", name="RandomFlip"),
                layers.RandomRotation(0.2, name="RandomRotation"),
                layers.RandomTranslation(
                    height_factor=(-0.2, 0.3),
                    width_factor=(-0.2, 0.3),
                    name="RandomTranslation",
                ),
            ],
            name="Augmentations",
        )(inputs)

        # Rescale the pixels to have values between 0 and 1
        x = layers.Rescaling(1.0 / 255, name="Rescaling")(x)

    else:
        # Rescale the pixels to have values between 0 and 1
        x = layers.Rescaling(1.0 / 255, name="Rescaling")(inputs)

    # Run in inference mode to be safe when we unfreeze the base model for fine-tuning
    x = mobilenet_layers(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(rate=0.2, name="Dropout")(x)

    # TODO @joffreylgt: set value in configuration
    #  https://github.com/JoffreyLGT/e-commerce-mlops/issues/103
    nb_output_classes = 27

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoint_file_path = (
        checkpoints_dir / "cp_loss-{val_loss:.2f}_acc-{val_accuracy:.2f}.ckpt"
    )
    history_file_path = args.output_dir / "history.csv"
    tensorboard_logs_dir = args.output_dir / "tensorboard_logs"
    outputs = layers.Dense(nb_output_classes, name="Output")(x)
    model = tf.keras.Model(inputs, outputs, name="RakutenImageNet")

    model.build((None, model_settings.IMG_WIDTH, model_settings.IMG_HEIGHT, 3))
    model.compile(
        optimizer=SGD(learning_rate=0.005, momentum=0.9),
        loss=CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
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
        TensorBoard(log_dir=str(tensorboard_logs_dir), histogram_freq=1),
    ]

    logger.info("Start model training")
    steps_per_epoch = nb_train_images // args.batch_size
    validation_steps = nb_test_images // args.batch_size

    # Train the top layer
    model.fit(
        train_ds.repeat(),
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_ds.repeat(),
        validation_steps=validation_steps,
        callbacks=cp_callbacks,
    )

    logger.info("Start model fine tuning")
    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    mobilenet_layers.trainable = True
    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(
        train_ds.repeat(),
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_ds.repeat(),
        validation_steps=validation_steps,
        callbacks=cp_callbacks,
    )

    logger.info("Save the model")
    # Load the latest checkpoint to avoid overfitting
    latest = latest_checkpoint(checkpoints_dir)
    if latest is not None:
        model.load_weights(latest)
    model.save(args.output_dir / "cnn_mobilenetv2.keras")

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
        model=TrainImageModelArgs,
        description=(
            "Create and train a new image model "
            "using dataset provided with --input-dir, "
            "then save it to --output-dir with "
            "its performance statistics."
        ),
    )
    args = parser.parse_typed_args()
    sys.exit(main(args))
