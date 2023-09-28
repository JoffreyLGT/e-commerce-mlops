"""Train fusion model, evaluate its performance and generates figures and stats.

Training is done using results from both text and image models without their
classification layer.

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
import itertools
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import keras
import mlflow
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
from keras.optimizers.legacy import SGD
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.types.schema import ColSpec, Schema
from pydantic import BaseModel, DirectoryPath
from sklearn import metrics
from tensorflow import train

from scripts.train_image_model import ImageClassificationWrapper
from scripts.train_text_model import TextClassificationWrapper
from src.core import constants
from src.core.settings import get_training_settings
from src.transformers.text_transformer import TextPreprocess
from src.utilities.dataset_utils import (
    get_empty_product_category,
    get_imgs_filenames,
    get_probabilities_header,
    get_product_category_probabilities,
    to_img_feature_target,
    to_normal_category_id,
    to_simplified_category_id,  # pyright: ignore
)
from src.utilities.mlflow_utils import set_staging_stage, setup_mlflow
from src.utilities.model_utils import (
    gen_classification_report,
    gen_confusion_matrix,
    gen_training_history_figure,
    generate_requirements,
)


class FusionClassificationWrapper(PythonModel):  # type: ignore
    """Fusion classification model with image and text model."""

    def load_context(self, context: PythonModelContext) -> None:
        """Called when the wrapper is created, load model.

        Args:
            context: provided automatically, contains MLFlow model information.
        """
        self.logger = logging.getLogger("FusionClassificationWrapper")
        self.fusion_model = tf.keras.models.load_model(
            context.artifacts["fusion_model"]
        )
        self.image_model = tf.keras.models.load_model(context.artifacts["image_model"])
        self.headless_image_model = keras.Model(
            inputs=self.image_model.input, outputs=self.image_model.layers[-2].output
        )
        self.text_model = tf.keras.models.load_model(context.artifacts["text_model"])
        vocabulary = None
        idfs = None
        with Path(context.artifacts["vocabulary"]).open("r") as file:
            vocabulary = json.load(file)
        with Path(context.artifacts["idfs"]).open("r") as file:
            idfs = np.array(json.load(file))
        self.text_preprocessor = TextPreprocess(vocabulary, idfs)
        self.headless_text_model = keras.Model(
            inputs=self.text_model.input, outputs=self.text_model.layers[-2].output
        )

    def predict_text_only(
        self, model_input: pd.DataFrame
    ) -> list[Sequence[str | int | float]]:
        """Predict product category using text model."""
        if len(model_input.index) == 0:
            return list()
        text = model_input["designation"] + " " + model_input["description"]
        preprocessed = self.text_preprocessor.transform(text)
        text_ds = tf.data.Dataset.from_tensor_slices(
            preprocessed  # type: ignore
        ).batch(96)
        y_predicted = self.text_model.predict(text_ds)
        return get_product_category_probabilities(
            model_input["product_id"].values, y_predicted, False
        )

    def predict_images_only(
        self, model_input: pd.DataFrame
    ) -> list[Sequence[str | int | float]]:
        """Predict product category using image model."""
        if len(model_input.index) == 0:
            return list()
        image_ds = (
            tf.data.Dataset.from_tensor_slices(model_input["image_path"].to_numpy())
            .map(to_img_feature_target)
            .batch(96)
        )
        y_predicted = self.image_model.predict(image_ds)
        return get_product_category_probabilities(
            model_input["product_id"].values, y_predicted, False
        )

    def predict_fusion(
        self, model_input: pd.DataFrame
    ) -> list[Sequence[str | int | float]]:
        """Predict product category using text model."""
        if len(model_input.index) == 0:
            return list()

        text_data = model_input["designation"] + " " + model_input["description"]
        preprocessed = self.text_preprocessor.transform(text_data)
        text_ds = tf.data.Dataset.from_tensor_slices(
            preprocessed  # type: ignore
        ).batch(96)
        text_output = self.headless_text_model.predict(text_ds)
        image_ds = (
            tf.data.Dataset.from_tensor_slices(model_input["image_path"].to_numpy())
            .map(to_img_feature_target)
            .batch(96)
        )
        images_output = self.headless_image_model.predict(image_ds)

        fusion_data = np.concatenate([text_output, images_output], axis=1)
        fusion_ds = tf.data.Dataset.from_tensor_slices(fusion_data).batch(96)
        y_predicted = self.fusion_model.predict(fusion_ds)
        return get_product_category_probabilities(
            model_input["product_id"].to_numpy(), y_predicted, False
        )

    def predict(
        self,
        context: PythonModelContext,  # pyright: ignore
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,  # pyright: ignore
    ) -> pd.DataFrame:
        """Predict product category."""
        text_only = model_input[
            ((model_input["designation"] != "") | (model_input["description"] != ""))
            & (model_input["image_path"] == "")
        ]

        images_only = model_input[
            ((model_input["designation"] == "") & (model_input["description"] == ""))
            & (model_input["image_path"] != "")
        ]

        both = model_input[
            ((model_input["designation"] != "") | (model_input["description"] != ""))
            & (model_input["image_path"] != "")
        ]

        error = model_input[
            ((model_input["designation"] == "") | (model_input["description"] == ""))
            & (model_input["image_path"] == "")
        ]

        y_text = self.predict_text_only(text_only)
        y_images = self.predict_images_only(images_only)
        y_fusion = self.predict_fusion(both)
        y_error = get_empty_product_category(error["product_id"].to_numpy(), False)

        data = np.array(list(itertools.chain(y_text, y_images, y_fusion, y_error)))
        return pd.DataFrame(data=data, columns=np.array(get_probabilities_header()))


class SavedModelPaths(NamedTuple):
    """Object containing path of all artifacts to save."""

    text_path: str
    vocabulary_path: str
    idfs_path: str
    image_path: str
    fusion_path: str


def save_models(
    output_dir: Path | str,
    text_model: keras.Model,
    image_model: keras.Model,
    fusion_model: keras.Model,
    text_preprocessor: TextPreprocess,
) -> SavedModelPaths:
    """Save all models in output_dir and return their uri.

    Args:
        output_dir: dir to save the models.
        text_model: text model to save.
        image_model: image model to save.
        fusion_model: fusion model to save.
        text_preprocessor: preprocessor to pickle and save.

    Returns:
        All uris into an object.
    """
    settings = get_training_settings()
    paths = SavedModelPaths(
        text_path=str((Path(output_dir) / f"{text_model.name}.tf").absolute()),
        vocabulary_path=str(Path(output_dir) / settings.TEXT_VOCABULARY_FILE_NAME),
        idfs_path=str(Path(output_dir) / settings.TEXT_IDFS_FILE_NAME),
        image_path=str((Path(output_dir) / f"{image_model.name}.tf").absolute()),
        fusion_path=str((Path(output_dir) / f"{fusion_model.name}.tf").absolute()),
    )
    text_model.save(paths.text_path)
    text_preprocessor.save_voc(paths.vocabulary_path, paths.idfs_path)
    image_model.save(paths.image_path)
    fusion_model.save(paths.fusion_path)

    return paths


def log_model_wrapper(
    models_paths: SavedModelPaths,
    requirements_path: str,
) -> ModelInfo:
    """Create a model wrapper with its schema and log it to mlflow."""
    input_schema = Schema(
        [
            ColSpec("string", "product_id"),
            ColSpec("string", "designation"),
            ColSpec("string", "description"),
            ColSpec("string", "image_path"),
        ]
    )
    output_cols = [
        ColSpec("float", f"{category_id}")
        for category_id in sorted(constants.CATEGORIES_DIC.keys())
    ]
    output_schema = Schema([ColSpec("string", "product_id"), *output_cols])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    return mlflow.pyfunc.log_model(
        artifact_path="fusion-model",
        python_model=FusionClassificationWrapper(),
        artifacts={
            "fusion_model": models_paths.fusion_path,
            "image_model": models_paths.image_path,
            "text_model": models_paths.text_path,
            "vocabulary": models_paths.vocabulary_path,
            "idfs": models_paths.idfs_path,
        },
        code_path=["src", "scripts"],
        signature=signature,
        pip_requirements=requirements_path,
    )


class TrainFusionModelArgs(BaseModel):
    """Hold all scripts arguments and do type checking."""

    input_dir: DirectoryPath = pydantic.Field(
        description="Directory containing the datasets to use."
    )
    output_dir: Path = pydantic.Field(
        description="Directory to save trained model and stats."
    )
    text_model_uri: str = pydantic.Field(
        "models:/text/Staging", description="URI to the text model logged with MLFlow."
    )
    image_model_uri: str = pydantic.Field(
        "models:/image/Staging",
        description="URI to the image model logged with MLFlow.",
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
        "fusion",
        descriptions=(
            "Define registered model name in MLFlow. "
            "Used only with set-staging flag."
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
    logger = logging.getLogger(__file__)
    logger.info("Script started")
    logger.info(f"Args: {args}")

    fusion_model_name = "Fusion_PD"

    setup_mlflow(fusion_model_name, {"type": "fusion"})
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

    X_train_path = (args.input_dir / "X_train.csv").absolute()
    X_test_path = (args.input_dir / "X_test.csv").absolute()
    df_X_train = pd.read_csv(X_train_path, index_col=0).fillna("")
    df_X_test = pd.read_csv(X_test_path, index_col=0).fillna("")

    mlflow.log_input(
        from_pandas(df_X_train, source=str(X_train_path), name="X_train"),
        "training",
    )
    mlflow.log_input(
        from_pandas(df_X_test, source=str(X_test_path), name="X_test"),
        "testing",
    )

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

    nb_train = len(X_train_text)
    nb_test = len(X_test_text)
    mlflow.log_metrics({"nb_train": nb_train, "nb_test": nb_test})

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

    logger.info("Load text model and text preprocessor")
    wrapped_text_model: TextClassificationWrapper = mlflow.pyfunc.load_model(
        args.text_model_uri
    ).unwrap_python_model()
    preprocessor = wrapped_text_model.preprocessor
    X_train_text_tensor = preprocessor.transform(X_train_text)
    X_test_text_tensor = preprocessor.transform(X_test_text)

    logger.info("Create text datasets")
    train_txt_ds = tf.data.Dataset.from_tensor_slices(
        (X_train_text_tensor, y_train_categorical)  # type: ignore
    ).batch(args.batch_size)
    test_txt_ds = tf.data.Dataset.from_tensor_slices(
        (X_test_text_tensor, y_test_categorical)  # type: ignore
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

    logger.info("Remove classifiation layer from text model")
    text_model = wrapped_text_model.model

    headless_text_model = tf.keras.Model(
        name=f"Headless_{text_model.name}",
        inputs=text_model.inputs,
        outputs=text_model.layers[-2].output,
    )
    headless_text_model.summary()

    logger.info("Load image model without classification layers")
    wrapped_image_model: ImageClassificationWrapper = mlflow.pyfunc.load_model(
        args.image_model_uri
    ).unwrap_python_model()
    image_model = wrapped_image_model.tf_model
    headless_image_model = tf.keras.Model(
        name=f"Headless_{image_model.name}",
        inputs=image_model.inputs,
        outputs=image_model.layers[-2].output,
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
        checkpoints_dir / "{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}-ckpt"
    )
    history_file_path = args.output_dir / "history.csv"
    tensorboard_logs_dir = args.output_dir / "tensorboard_logs"

    logger.info("Build fusion model")
    fusion_model = tf.keras.models.Sequential(name=fusion_model_name)
    fusion_model.add(layers.InputLayer(input_shape=(train_fusion.shape[1])))
    fusion_model.add(layers.Dense(units=512, activation="relu"))
    fusion_model.add(layers.Dropout(rate=0.2))

    fusion_model.add(layers.Dense(units=128, activation="relu"))
    fusion_model.add(layers.Dropout(rate=0.2))

    fusion_model.add(
        layers.Dense(
            len(constants.CATEGORIES_DIC.keys()), activation="softmax", name="Output"
        )
    )

    fusion_model.build((None, train_fusion.shape[1]))
    fusion_model.compile(
        optimizer=SGD(learning_rate=0.005, momentum=0.9),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    fusion_model.summary()

    initial_epoch: int = 1
    latest = train.latest_checkpoint(checkpoints_dir)
    if latest is not None:
        logger.info(f"Load checkpoint: {latest}")
        fusion_model.load_weights(latest)
        initial_epoch = int(Path(latest).name.split("-")[0])
    else:
        logger.info("No checkpoint to load")

    logger.info("Evaluate initial val_loss and val_accuracy")
    _, val_accuracy = fusion_model.evaluate(test_fusion_ds)

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

    logger.info("Start fusion model training")
    fusion_model.fit(
        train_fusion_ds,
        epochs=initial_epoch + args.epochs,
        validation_data=test_fusion_ds,
        callbacks=cp_callbacks,
        initial_epoch=initial_epoch,
    )

    logger.info("Save the fusion_model")
    # Load the latest checkpoint to avoid overfitting
    latest = train.latest_checkpoint(checkpoints_dir)
    if latest is not None:
        fusion_model.load_weights(latest)

    logger.info("Save all models")
    models_paths = save_models(
        args.output_dir,
        text_model=text_model,
        image_model=image_model,
        fusion_model=fusion_model,
        text_preprocessor=preprocessor,
    )

    logger.info("Create model wrapper and save send it to MLFlow")
    model_info = log_model_wrapper(
        models_paths,
        requirements_path=str(requirements_file_path),
    )
    if args.set_staging:
        logger.info("Set model status to staging")
        set_staging_stage(
            model_info, args.registered_model, tags={"name": fusion_model_name}
        )

    logger.info("Generate training history figure")
    history_fig_path = str(args.output_dir / settings.TRAINING_HISTORY_FILE_NAME)
    logger.info(gen_training_history_figure(history_file_path, history_fig_path))
    mlflow.log_artifact(history_fig_path)

    logger.info("Predict test data categories")
    y_pred_simplified = fusion_model.predict(test_fusion_ds)
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
