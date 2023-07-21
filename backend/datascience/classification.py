import os
import pickle
from typing import List, Tuple
import pandas as pd
import numpy as np
from tensorflow import keras
import tensorflow as tf
from datascience.src.data import (
    CATEGORIES_DIC,
    PRDTYPECODE_DIC,
    convert_sparse_matrix_to_sparse_tensor,
)

PICKLES_DIR = os.path.join("datascience", "data", "pickles")
MODELS_DIR = os.path.join("datascience", "data", "models")


def get_prediction(
    designation: str, description: str, image: np.ndarray
) -> List[Tuple[int, float, str]]:
    # Get the predictions from model
    predictions = predict_prdtypecode(designation, description, image)
    # Get the top best prediction, which in at the first position
    # since the predictions are sorted by probabilities descending
    top_3 = [(i[0], i[1], CATEGORIES_DIC[i[0]]) for i in predictions[0][0:3]]
    return top_3


def preprocess_image(img):
    return tf.image.resize(img, [224, 224])


def predict_prdtypecode(designation: str, description: str, image: np.ndarray):
    # Load TextPreprocessor already fitted on training data
    with open(os.path.join(PICKLES_DIR, "text_preprocessor.pkl"), "rb") as fp:
        text_preprocessor = pickle.load(fp)

    # Load text model
    text_model = keras.models.load_model(
        os.path.join(MODELS_DIR, "text", "mlp_model_v2.h5")  # , compile=False
    )
    text_model_wo_head = tf.keras.Model(
        inputs=text_model.inputs, outputs=text_model.layers[-2].output
    )

    # Load image model
    image_model = keras.models.load_model(
        os.path.join(MODELS_DIR, "image", "cnn_mobilenetv2.h5"), compile=False
    )
    image_model_wo_head = tf.keras.Model(
        inputs=image_model.inputs, outputs=image_model.layers[-2].output
    )

    # Load fusion model
    fusion_model = keras.models.load_model(
        os.path.join(MODELS_DIR, "fusion", "fusion_text_image.h5")
    )

    # Data preprocessing
    feats = pd.Series(f"{designation} {description}")
    feats_processed = text_preprocessor.transform(feats)

    # Predict prdtypecode on text model
    text_predictions = text_model_wo_head.predict(
        convert_sparse_matrix_to_sparse_tensor(feats_processed)
    )

    # Predict prdtypecode on image model
    img_dataset = (
        tf.data.Dataset.from_tensor_slices([image]).map(preprocess_image).batch(1)
    )
    image_predictions = image_model_wo_head.predict(img_dataset)

    # Concatenate both results
    concat_predictions = np.concatenate((text_predictions, image_predictions), axis=1)

    fusion_dataset = tf.data.Dataset.from_tensor_slices(concat_predictions).batch(1)
    y_pred = fusion_model.predict(fusion_dataset)
    return get_prdtypecode_probabilities(y_pred)


def get_prdtypecode_probabilities(y_pred):
    """
    Get the probabilities for each categories from the predictions.

    Argument:
    - y_pred: predictions from the model

    Returns:
    - a list of [prdtypecode, probability in percent] sorted descending
    """
    list_decisions = []
    for y in y_pred:
        list_probabilities = []
        for i, probability in enumerate(y):
            list_probabilities.append(
                [list(PRDTYPECODE_DIC.keys())[i], np.round(probability * 100, 2)]
            )
        list_decisions.append(
            sorted(list_probabilities, key=lambda x: x[1], reverse=True)
        )
    return list_decisions
