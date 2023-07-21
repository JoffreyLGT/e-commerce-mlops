# pylint: disable=no-name-in-module

import os

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.layers import Dropout

CHECKPOINT_DIR = os.path.join("datascience", "data", "models", "text_checkpoints")
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, "cp_{val_loss:.2f}-{val_accuracy:.2f}-.ckpt"
)

# Path to the history CSV file to store training metrics
HIST_CSV_PATH = os.path.join(CHECKPOINT_DIR, "history.csv")


def get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    # Arguments
        num_classes: int, number of classes.

    # Returns
        units, activation values.
    """
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    return units, activation


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    op_units, op_activation = get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(InputLayer(input_shape=(input_shape), sparse=True))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation="relu"))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model


def train_ngram_model(
    data,
    num_classes,
    learning_rate=1e-3,
    epochs=1000,
    batch_size=128,
    layers=2,
    units=128,  # 64,
    dropout_rate=0.2,
):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (train_dataset, train_labels), (val_dataset, val_labels) = data

    # Vectorize texts.
    x_train, x_val = train_dataset, val_dataset

    # Create model instance.
    model = mlp_model(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1],
        num_classes=num_classes,
    )

    loss = "categorical_crossentropy"
    optimizer = Adam(lr=learning_rate)

    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])  # type: ignore

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=1, restore_best_weights=True
        ),
        # start_from_epoch = 5,
        # Insert the metrics into a CSV file)
        # tf.keras.callbacks.CSVLogger(HIST_CSV_PATH, separator=",", append=False),
    ]

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Train and validate model.
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        # verbose=2,  # Logs once per epoch.
        batch_size=batch_size,
    )

    # Print results.
    history = history.history  # type: ignore
    print(
        f"Validation accuracy: {history['val_acc'][-1]}, loss: {history['val_loss'][-1]}"
    )

    # Save model.
    model_destination = os.path.join(
        "datascience", "data", "models", "text", "mlp_model_v2.h5"
    )
    model.save(model_destination)
    return history["val_acc"][-1], history["val_loss"][-1]
