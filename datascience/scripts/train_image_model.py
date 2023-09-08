# type: ignore
"""Train image model, evaluate its performance and generates figures and stats.

from shutil import ignore_patterns


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
import itertools
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import pydantic_argparse
import tensorflow as tf
from pydantic import BaseModel, DirectoryPath
from sklearn import metrics
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

from src.core.settings import get_mobilenet_image_model_settings


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

    model_settings = get_mobilenet_image_model_settings()

    logger.info("Create datasets")

    train_dir = args.input_dir / "images" / "train"
    test_dir = args.input_dir / "images" / "test"
    (train_ds, val_ds) = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="both",
        seed=args.seed,
        image_size=(model_settings.IMG_WIDTH, model_settings.IMG_HEIGHT),
        batch_size=args.batch_size,
        label_mode="categorical",
    )

    nb_train_images = len(train_ds.file_paths)
    nb_val_images = len(val_ds.file_paths)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(model_settings.IMG_WIDTH, model_settings.IMG_HEIGHT),
        batch_size=args.batch_size,
        label_mode="categorical",
    )
    # Add cache configuration to speed up training
    # If this cause issues, we'll add an argument to enable it
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

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
        TensorBoard(log_dir=tensorboard_logs_dir, histogram_freq=1),
    ]

    logger.info("Start model training")
    steps_per_epoch = nb_train_images // args.batch_size
    validation_steps = nb_val_images // args.batch_size

    # Train the top layer
    model.fit(
        train_ds.repeat(),
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds.repeat(),
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
        loss=CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(
        train_ds.repeat(),
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds.repeat(),
        validation_steps=validation_steps,
        callbacks=cp_callbacks,
    )

    logger.info("Save the model")
    # Load the latest checkpoint to avoid overfitting
    latest = latest_checkpoint(checkpoints_dir)
    if latest is not None:
        model.load_weights(latest)
    model.save(args.output_dir / "cnn_mobilenetv2.keras")

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

    logger.info("Predict category on test data")

    predictions = np.array([])
    labels = np.array([])
    for x, y in test_ds:
        predictions = np.concatenate(
            [predictions, np.argmax(model.predict(x, verbose=0), axis=-1)]
        )
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    prdtypecodes = sorted([int(x) for x in os.listdir(test_dir)])

    test_labels = [prdtypecodes[int(i)] for i in labels]
    predictions_labels = [prdtypecodes[int(i)] for i in predictions]

    logger.info(
        f"Accuracy score: {metrics.accuracy_score(test_labels, predictions_labels)}"
    )
    logger.info("Generate classification report")
    class_report = metrics.classification_report(
        test_labels, predictions_labels, zero_division=0.0  # pyright: ignore
    )

    Path(args.output_dir / "classification_report.txt").write_text(str(class_report))
    logger.info(str(class_report))

    logger.info("Generate confusion matrix")
    cnf_matrix = np.round(
        metrics.confusion_matrix(test_labels, predictions_labels, normalize="true"), 2
    )

    classes = range(0, nb_output_classes)

    plt.figure(figsize=(13, 13))

    plt.imshow(cnf_matrix, interpolation="nearest", cmap="Blues")
    plt.title("Matrice de confusion")
    tick_marks = classes
    plt.xticks(tick_marks, prdtypecodes)
    plt.yticks(tick_marks, prdtypecodes)

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

    plt.ylabel("Vrais labels")
    plt.xlabel("Labels prédits")
    plt.xticks(rotation=45)
    plt.savefig(args.output_dir / "confusion_matrix.png")

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
