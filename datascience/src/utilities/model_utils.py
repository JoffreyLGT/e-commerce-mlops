"""Utility functions to evaluate models performance and generate figures."""
import itertools
import subprocess
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import FilePath
from sklearn import metrics

from src.core import constants
from src.core.custom_errors import RequirementsGenerationError


def gen_training_history_figure(history_file_path: FilePath, output_file: str) -> str:
    """Generate a figure with training history data.

    One subfigure shows training and validation accuracy.
    A second subfigure shows training and validation loss.

    Args:
        history_file_path: path to the csv containing training history.
        output_file: file path to save the figure in.

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
    file_path = output_file
    plt.savefig(file_path)
    return file_path


def gen_classification_report(
    y: Iterable[int],
    y_pred: Iterable[int],
    output_file: str,
) -> str:
    """Generate a classification report and save it into a txt file.

    Args:
        y: target values.
        y_pred: predicted values.
        output_file: path to save the txt file in.

    Returns:
        Classification report.
    """
    class_report = str(
        metrics.classification_report(y, y_pred, zero_division=0.0)  # pyright: ignore
    )
    file_path = Path(output_file)
    file_path.write_text(class_report)
    return class_report


def gen_confusion_matrix(
    y: Iterable[int],
    y_pred: Iterable[int],
    output_file: str,
) -> None:
    """Generate a confusion matrix and save the figure.

    Args:
        y: target values.
        y_pred: predicted values.
        output_file: path to save the txt file in.
    """
    cnf_matrix = np.round(metrics.confusion_matrix(y, y_pred, normalize="true"), 2)

    classes = range(0, len(constants.CATEGORIES_DIC.keys()))
    category_ids = list(constants.CATEGORIES_DIC.keys())

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
    plt.savefig(output_file)


def generate_requirements(file_path: Path | str) -> None:
    """Generate the requirements using Poetry.

    Args:
        file_path: where to save the file.
    """
    try:
        subprocess.run(
            [
                "poetry",
                "export",
                "--without",
                "dev",
                "--without-hashes",
                "-f",
                "requirements.txt",
                "-o",
                f"{file_path}",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as ex:
        raise RequirementsGenerationError from ex
