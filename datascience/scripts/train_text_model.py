"""Train text model, evaluate its performance and generates figures and stats.

All logs and best checkpoints are stored in --output-dir.
--input-dir expects a dataset with this structure is expected:

dataset_dir
├── X_test.csv
├── X_train.csv
├── images
│   ├── image_977803476_product_278535420.jpg
│   └── image_1174586892_product_2940638801.jpg
├── y_test.csv
└── y_train.csv

Use scripts/create_datasets.py to generate a dataset.
"""
import json
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
from keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from keras.layers import Dense, Dropout, InputLayer
from keras.losses import CategoricalCrossentropy
from keras.optimizers.legacy import (
    Adam,
)
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import ModelSignature, PythonModel, PythonModelContext
from mlflow.types import ColSpec, Schema
from pydantic import BaseModel, DirectoryPath
from sklearn import metrics
from tensorflow import train

from src.core import constants
from src.core.settings import (
    get_training_settings,
)
from src.transformers.text_transformer import (
    TextPreprocess,
)
from src.utilities.dataset_utils import (
    get_product_category_probabilities,
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


class TextClassificationWrapper(PythonModel):  # type: ignore
    """Text classification model with preprocessor and output conversion."""

    def load_context(self, context: PythonModelContext) -> None:
        """Called when the wrapper is created, load model.

        Args:
            context: provided automatically, contains MLFlow model information.
        """
        self.model = tf.keras.models.load_model(context.artifacts["model"])
        vocabulary = None
        idfs = None
        with Path(context.artifacts["vocabulary"]).open("r") as file:
            vocabulary = json.load(file)
        with Path(context.artifacts["idfs"]).open("r") as file:
            idfs = json.load(file)
        self.preprocessor = TextPreprocess(vocabulary, idfs)

    def predict(
        self,
        context: PythonModelContext,  # pyright: ignore
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,  # pyright: ignore
    ) -> list[Sequence[str | int | float]]:
        """Predict product category."""
        values = model_input["designation"] + " " + model_input["description"]
        preprocessed_values = self.preprocessor.transform(values)
        predict_ds = tf.data.Dataset.from_tensor_slices(
            preprocessed_values  # type: ignore
        ).batch(96)
        y_predicted = self.model.predict(predict_ds)
        return get_product_category_probabilities(
            model_input["product_id"].values, y_predicted, True
        )


def log_model_wrapper(
    artifact_path: str,
    keras_model_path: str,
    vocabulary_path: str,
    idfs_path: str,
    requirements_path: str,
) -> ModelInfo:
    """Create a model wrapper with its schema and log it to mlflow."""
    input_schema = Schema(
        [
            ColSpec("string", "product_id"),
            ColSpec("string", "designation"),
            ColSpec("string", "description"),
        ]
    )
    output_cols = [
        ColSpec("float", f"{category_id}")
        for category_id in sorted(constants.CATEGORIES_DIC.keys())
    ]
    output_schema = Schema([ColSpec("string", "product_id"), *output_cols])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    return mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=TextClassificationWrapper(),
        artifacts={
            "model": keras_model_path,
            "vocabulary": vocabulary_path,
            "idfs": idfs_path,
        },
        code_path=["src", "scripts"],
        signature=signature,
        pip_requirements=requirements_path,
    )


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
        "text",
        descriptions=(
            "Define registered model name in MLFlow. "
            "Used only with set-staging flag."
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
    logger = logging.getLogger(__file__)
    logger.info("Script started")
    logger.info(f"Args: {args}")

    model_name = "Text_PD_MLP"
    setup_mlflow(model_name, {"type": "text"})
    mlflow.tensorflow.autolog(log_datasets=False, log_models=False)

    settings = get_training_settings()
    mlflow.log_param("args", args.dict())

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    logger.info("Extract features")
    X_train = df_X_train["designation"] + " " + df_X_train["description"]
    X_test = df_X_test["designation"] + " " + df_X_test["description"]

    logger.info("Extract targets")
    y_train_path = args.input_dir / "y_train.csv"
    y_train = np.loadtxt(
        y_train_path,
        dtype=int,
        delimiter=",",
        skiprows=1,
        usecols=(1),
    )
    y_test_path = args.input_dir / "y_test.csv"
    y_test = np.loadtxt(
        y_test_path,
        dtype=int,
        delimiter=",",
        skiprows=1,
        usecols=(1),
    )
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

    logger.info("Create preprocessor and preprocess features")
    preprocessor = TextPreprocess()
    X_train_tensor = preprocessor.fit_transform(X_train)
    X_test_tensor = preprocessor.transform(X_test)

    vocabulary_path = args.output_dir / settings.TEXT_VOCABULARY_FILE_NAME
    idfs_path = args.output_dir / settings.TEXT_IDFS_FILE_NAME
    preprocessor.save_voc(vocabulary_path, idfs_path)
    logger.info(f"Vocabulary saved: {vocabulary_path}")
    logger.info(f"Idfs saved: {idfs_path}")

    logger.info("Create datasets")
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train_tensor, y_train_categorical)  # type: ignore
    ).batch(args.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (X_test_tensor, y_test_categorical)  # type: ignore
    ).batch(args.batch_size)

    nb_train = len(X_train)
    nb_test = len(X_test)
    mlflow.log_metrics({"nb_train": nb_train, "nb_test": nb_test})
    # # Add cache configuration to speed up training
    # # If this cause issues, we'll add an argument to enable it
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    logger.info("Build text model")
    model = tf.keras.models.Sequential(name=model_name)
    model.add(InputLayer(input_shape=(X_train_tensor.shape[1]), sparse=True))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(rate=0.2))

    nb_output_classes = len(constants.CATEGORIES_DIC.keys())

    model.add(Dense(units=nb_output_classes, activation="softmax"))

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoint_file_path = (
        checkpoints_dir / "{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}-ckpt"
    )
    history_file_path = args.output_dir / "history.csv"
    tensorboard_logs_dir = args.output_dir / "tensorboard_logs"

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
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
    model.fit(
        train_ds,
        epochs=initial_epoch + args.epochs,
        validation_data=test_ds,
        callbacks=cp_callbacks,
        initial_epoch=initial_epoch,
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
        artifact_path="text-model",
        keras_model_path=str(model_file_path),
        vocabulary_path=str(vocabulary_path),
        idfs_path=str(idfs_path),
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
