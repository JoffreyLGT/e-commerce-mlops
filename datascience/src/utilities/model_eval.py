"""Utility functions to evaluate models performance and generate figures."""
import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import DirectoryPath, FilePath
from sklearn import metrics

from src.core.settings import get_common_settings


def gen_training_history_figure(
    history_file_path: FilePath, output_dir: DirectoryPath
) -> Path:
    """Generate a figure with training history data.

    One subfigure shows training and validation accuracy.
    A second subfigure shows training and validation loss.

    Args:
        history_file_path: path to the csv containing training history.
        output_dir: directory to save the figure in.

    Returns:
        Path to the generated figure.
    """
    training_history = pd.read_csv(history_file_path, delimiter=",", header=0)

    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(121)

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")

    ax1.plot(
        np.arange(1, training_history["accuracy"].count() + 1, 1),
        training_history["accuracy"],
        label="Training Accuracy",
        color="blue",
    )

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
    file_path = output_dir / "training_history.png"
    plt.savefig(file_path)
    return file_path


def gen_classification_report(
    y: np.ndarray[Any, np.dtype[np.int32]],
    y_pred: np.ndarray[Any, np.dtype[np.int32]],
    output_dir: DirectoryPath,
) -> tuple[Path, str]:
    """Generate a classification report and save it into a txt file.

    Args:
        y: target values.
        y_pred: predicted values.
        output_dir: directory to save the txt file in.

    Returns:
        A tuple with (path to savex txt file, classification report).
    """
    class_report = str(
        metrics.classification_report(y, y_pred, zero_division=0.0)  # pyright: ignore
    )
    file_path = output_dir / "classification_report.txt"
    file_path.write_text(class_report)
    return (file_path, class_report)


def gen_confusion_matrix(
    y: np.ndarray[Any, np.dtype[np.int32]],
    y_pred: np.ndarray[Any, np.dtype[np.int32]],
    output_dir: DirectoryPath,
) -> Path:
    """Generate a confusion matrix and save the figure.

    Args:
        y: target values.
        y_pred: predicted values.
        output_dir: directory to save the txt file in.

    Returns:
        Path to the generated figure.
    """
    cnf_matrix = np.round(metrics.confusion_matrix(y, y_pred, normalize="true"), 2)

    settings = get_common_settings()
    classes = range(0, len(settings.CATEGORIES_DIC.keys()))
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
    file_path = output_dir / "confusion_matrix.png"
    plt.savefig(file_path)
    return file_path
