"""Contains all function to import models and get predictions."""


import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import Tensor, keras

from app.core.settings import backend_dir
from datascience.src.data import PRDTYPECODE_DIC, convert_sparse_matrix_to_sparse_tensor

PICKLES_DIR = os.path.join(backend_dir, "datascience", "data", "pickles")
MODELS_DIR = os.path.join(backend_dir, "datascience", "data", "models")


def preprocess_image(img: Tensor) -> Tensor:
    """Image preprocessing.

    Args:
        img: to process

    Returns:
        Tensor containing image data.
    """
    return tf.image.resize(img, [224, 224])


def predict_prdtypecode(
    designation: str | None,
    description: str | None,
    image: np.ndarray[Any, Any] | None,
) -> list[list[tuple[int, float]]]:
    """Use the models to predict the probabilities for the product to be part of each categories.

    Args:
        designation: short summary of the product
        description: full description of the product
        image: image of the product

    Returns:
        List of probabilities for the product to be from each categories.
    """
    has_text = designation is not None or description is not None
    has_image = image is not None
    has_both = has_text and has_image

    text_predictions = None
    image_predictions = None

    # Work with text model only if we have text data
    if has_text:
        # Load TextPreprocessor already fitted on training data
        with open(os.path.join(PICKLES_DIR, "text_preprocessor.pkl"), "rb") as file:
            text_preprocessor = pickle.load(file)

        # Load text model
        model_path = os.path.join(MODELS_DIR, "text", "mlp_model_v2.h5")
        text_model = keras.models.load_model(model_path)
        if text_model is None:
            raise TypeError(f"text_model loaded from {model_path} is None")

        # Data preprocessing
        feats = pd.Series(f"{designation} {description}")
        feats_processed = text_preprocessor.transform(feats)

        if has_both:
            text_model = tf.keras.Model(
                inputs=text_model.inputs, outputs=text_model.layers[-2].output
            )
            # Predict prdtypecode on text model
            text_predictions = text_model.predict(
                convert_sparse_matrix_to_sparse_tensor(feats_processed)
            )
        else:
            text_predictions = text_model.predict(
                convert_sparse_matrix_to_sparse_tensor(feats_processed)
            )
            return get_prdtypecode_probabilities(text_predictions)

    # Work with image model only if we have image data
    if has_image and image is not None:
        # Load image model
        model_path = os.path.join(MODELS_DIR, "image", "cnn_mobilenetv2.h5")
        image_model = keras.models.load_model(model_path, compile=False)
        if image_model is None:
            raise TypeError(f"image_model loaded from {model_path} is None")

        img_dataset = (
            tf.data.Dataset.from_tensor_slices([image]).map(preprocess_image).batch(1)
        )

        if has_both:
            image_model = tf.keras.Model(
                inputs=image_model.inputs, outputs=image_model.layers[-2].output
            )
            # Predict prdtypecode on image model
            image_predictions = image_model.predict(img_dataset)
        else:
            return get_prdtypecode_probabilities(image_model.predict(img_dataset))

    # Load fusion model
    model_path = os.path.join(MODELS_DIR, "fusion", "fusion_text_image.h5")
    fusion_model = keras.models.load_model(model_path)
    if fusion_model is None:
        raise TypeError(f"fusion_model loaded from {model_path} is None")

    # Create dataset
    if text_predictions is None:
        raise TypeError("text_predictions shouldn't be None")
    if image_predictions is None:
        raise TypeError("image_predictions shouldn't be None")
    concat_predictions = np.concatenate((text_predictions, image_predictions), axis=1)
    fusion_dataset = tf.data.Dataset.from_tensor_slices(concat_predictions).batch(1)
    # Concatenate both results
    y_pred: list[list[float]] = fusion_model.predict(fusion_dataset)

    return get_prdtypecode_probabilities(y_pred)


def get_prdtypecode_probabilities(
    y_pred: list[list[float]],
) -> list[list[tuple[int, float]]]:
    """Get the probabilities for each categories from the predictions.

    Argument:
    - y_pred: predictions from the model

    Returns:
    - a list of [prdtypecode, probability in percent] sorted descending
    """
    list_decisions: list[list[tuple[int, float]]] = []
    for y in y_pred:
        list_probabilities: list[tuple[int, float]] = []
        for i, probability in enumerate(y):
            code = list(PRDTYPECODE_DIC.keys())
            if len(code) <= i:
                raise NotImplementedError(
                    "PRDTYPECODE_DIC does not contain {i} values."
                )
            list_probabilities.append(
                [code[i], np.around(probability * 100, 2)]  # type: ignore
            )
        list_decisions.append(
            sorted(list_probabilities, key=lambda x: x[1], reverse=True)
        )
    return list_decisions
