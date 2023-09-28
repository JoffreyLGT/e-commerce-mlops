"""Train image model, evaluate its performance and generates figures and stats.

All logs and best checkpoints are stored in --output-dir.
--input-dir expects a dataset with this structure is expected:

dataset_dir
├── X_test.csv
├── X_train.csv
├── images
│   └── image_977803476_product_278535420.jpg
│   └── image_1174586892_product_2940638801.jpg
├── y_test.csv
└── y_train.csv

Use scripts/create_datasets.py to generate a dataset.
"""
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import keras
import mlflow
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
from keras.optimizers.legacy import SGD
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import ModelSignature, PythonModel, PythonModelContext
from mlflow.types import ColSpec, Schema
from pydantic import BaseModel, DirectoryPath
from sklearn import metrics
from tensorflow import train

from src.core import constants
from src.core.settings import get_mobilenet_image_model_settings, get_training_settings
from src.utilities.dataset_utils import (
    get_imgs_filenames,
    get_product_category_probabilities,
    to_img_feature_target,
    to_normal_category_id,
    to_simplified_category_id,
)
from src.utilities.mlflow_utils import set_staging_stage, setup_mlflow
from src.utilities.model_utils import (
    gen_classification_report,
    gen_confusion_matrix,
    gen_training_history_figure,
    generate_requirements,
)


class ImageClassificationWrapper(PythonModel):  # type: ignore
    """Image classification model with preprocessor and output conversion."""

    tf_model: keras.Model

    def load_context(self, context: PythonModelContext) -> None:
        """Called when the wrapper is created, load model.

        Args:
            context: provided automatically, contains MLFlow model information.
        """
        self.tf_model = tf.keras.models.load_model(context.artifacts["model"])

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> list[Sequence[str | int | float]]:
        """Predict images category."""
        predict_ds = (
            tf.data.Dataset.from_tensor_slices(model_input["image_path"].to_numpy())
            .map(to_img_feature_target)
            .batch(96)
        )
        predictions = self.tf_model.predict(predict_ds)
        return get_product_category_probabilities(
            model_input["product_id"].to_numpy(), predictions, True
        )


def log_model_wrapper(
    artifact_path: str, keras_model_path: str, requirements_path: str
) -> ModelInfo:
    """Create a model wrapper with its schema and log it to mlflow."""
    input_schema = Schema(
        [ColSpec("string", "product_id"), ColSpec("string", "image_path")]
    )
    output_cols = [
        ColSpec("float", f"{category_id}")
        for category_id in sorted(constants.CATEGORIES_DIC.keys())
    ]
    output_schema = Schema([ColSpec("string", "product_id"), *output_cols])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    return mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=ImageClassificationWrapper(),
        artifacts={"model": keras_model_path},
        code_path=["src", "scripts"],
        signature=signature,
        pip_requirements=requirements_path,
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
    train_patience: int = pydantic.Field(
        10,
        description=(
            "Number of epoch to do without improvements "
            "before stopping the model training."
        ),
    )
    epochs: int = pydantic.Field(
        100,
        description=(
            "Number of epochs to reach before stopping."
            " Stops before if train-patience is reached."
        ),
    )
    set_staging: bool = pydantic.Field(
        False,
        description=("Set new model version status as staging for 'fusion' model"),
    )
    registered_model: str = pydantic.Field(
        "image",
        descriptions=(
            "Define registered model name in MLFlow. "
            "Used only with set-staging flag."
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
    logger.info(f"Args: {args}")

    model_name = "Image_PD_MobileNetV2"
    setup_mlflow(model_name, {"type": "image"})
    mlflow.tensorflow.autolog(log_datasets=False, log_models=False)

    mlflow.log_param("args", args.dict())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    settings = get_training_settings()
    requirements_file_path = args.output_dir / settings.REQUIREMENTS_FILE_NAME
    generate_requirements(requirements_file_path)

    # Ensure all messages from Tensorflow will be logged and not only printed
    tf.keras.utils.disable_interactive_logging()
    # Avoid Tensorflow flooding the console with messages
    # Especially annoying with tensorflow-metal
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    logger.info("Load data")

    X_train_path = str(args.input_dir / "X_train.csv")
    X_test_path = str(args.input_dir / "X_test.csv")
    df_X_train = pd.read_csv(X_train_path, index_col=0).fillna("")
    df_X_test = pd.read_csv(X_test_path, index_col=0).fillna("")

    mlflow.log_input(
        from_pandas(df_X_train, source=X_train_path, name="X_train"), "training"
    )
    mlflow.log_input(
        from_pandas(df_X_test, source=X_test_path, name="X_test"), "testing"
    )

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
    y_train_path = args.input_dir / "y_train.csv"
    y_train = np.loadtxt(
        y_train_path,
        dtype=int,
        delimiter=",",
        skiprows=1,
        usecols=(1),
    )
    y_test_path = args.input_dir / "y_test.csv"
    y_test = np.loadtxt(y_test_path, dtype=int, delimiter=",", skiprows=1, usecols=(1))
    mlflow.log_input(
        from_numpy(y_train, source=str(y_train_path), name="y_train"), "training"
    )
    mlflow.log_input(
        from_numpy(y_test, source=str(y_test_path), name="y_test"), "testing"
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
        tf.data.Dataset.from_tensor_slices((X_test.copy(), y_test_categorical))
        .map(to_img_feature_target)
        .batch(args.batch_size)
    )

    model_settings = get_mobilenet_image_model_settings()
    mlflow.log_param("mobilenet_image_model_settings", model_settings.dict())

    nb_train_images = len(X_train)
    nb_test_images = len(X_test)
    mlflow.log_metrics(
        {"nb_train_images": nb_train_images, "nb_test_images": nb_test_images}
    )

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

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoint_file_path = (
        checkpoints_dir / "{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.ckpt"
    )
    history_file_path = args.output_dir / "history.csv"
    tensorboard_logs_dir = args.output_dir / "tensorboard_logs"

    outputs = layers.Dense(
        len(constants.CATEGORIES_DIC.keys()), name="Output", activation="softmax"
    )(x)
    model = tf.keras.Model(inputs, outputs, name=model_name)

    model.build((None, model_settings.IMG_WIDTH, model_settings.IMG_HEIGHT, 3))
    model.compile(
        optimizer=SGD(learning_rate=0.005, momentum=0.9),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model.summary()

    initial_epoch: int = 1
    latest: str = train.latest_checkpoint(checkpoints_dir)
    if latest is not None:
        logger.info(f"Load checkpoint: {latest}")
        model.load_weights(latest)
        initial_epoch = int(Path(latest).name.split("-")[0])
    else:
        logger.info("No checkpoint to load")

    logger.info("Evaluate initial val_loss and val_accuracy")
    _, val_accuracy = model.evaluate(test_ds)

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
            initial_value_threshold=val_accuracy,
        ),
        # Insert the metrics into a CSV file
        CSVLogger(history_file_path, separator=",", append=True),
        # Log information to display them in TensorBoard
        TensorBoard(log_dir=str(tensorboard_logs_dir), histogram_freq=1),
    ]

    logger.info("Start model training")
    with mlflow.start_run(run_name="training", nested=True):
        # Train our custom layers
        history = model.fit(
            train_ds,
            epochs=initial_epoch + args.epochs,
            validation_data=test_ds,
            callbacks=cp_callbacks,
            initial_epoch=initial_epoch,
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
    with mlflow.start_run(run_name="fine-tuning", nested=True):
        # Train all the layers including MobileNet's
        model.fit(
            train_ds,
            epochs=initial_epoch + args.epochs,
            validation_data=test_ds,
            callbacks=cp_callbacks,
            initial_epoch=history.epoch[-1],
        )

    logger.info("Save the model")
    # Load the latest checkpoint to avoid overfitting
    latest = train.latest_checkpoint(checkpoints_dir)
    if latest is not None:
        model.load_weights(latest)
    model_file_path = args.output_dir / f"{model_name}.tf"
    model.save(model_file_path)

    logger.info("Create model wrapper and save send it to MLFlow")
    model_info = log_model_wrapper(
        artifact_path="image-model",
        keras_model_path=str(model_file_path),
        requirements_path=str(requirements_file_path),
    )

    if args.set_staging:
        logger.info("Set model status to staging")
        set_staging_stage(model_info, args.registered_model, tags={"name": model_name})

    logger.info("Generate training history figure")
    history_fig_path = str(args.output_dir / settings.TRAINING_HISTORY_FILE_NAME)
    logger.info(gen_training_history_figure(history_file_path, history_fig_path))
    mlflow.log_artifact(history_fig_path)

    logger.info("Predict test data categories")
    y_pred_simplified = model.predict(test_ds)
    y_pred = to_normal_category_id([int(np.argmax(i)) for i in y_pred_simplified])

    accuracy_score: float = float(metrics.accuracy_score(y_test, y_pred))
    logger.info(f"Accuracy score: {accuracy_score}")
    mlflow.log_metric("accuracy_score", accuracy_score)

    logger.info("Generate classification report")
    class_report_path = str(args.output_dir / settings.CLASSIFICATION_REPORT_FILE_NAME)
    class_report = gen_classification_report(y_test, y_pred, class_report_path)
    logger.info(class_report)
    mlflow.log_artifact(class_report_path)

    logger.info("Generate confusion matrix")
    confu_matrix_path = str(args.output_dir / settings.CONFUSION_MATRIX_FILE_NAME)
    gen_confusion_matrix(y_test, y_pred, confu_matrix_path)
    logger.info(confu_matrix_path)
    mlflow.log_artifact(confu_matrix_path)

    mlflow.end_run()
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
