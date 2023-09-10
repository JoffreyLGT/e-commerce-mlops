"""Train text model, evaluate its performance and generates figures and stats.
All logs and best checkpoints are stored in --output-dir.
"""

import os
from pathlib import Path

import data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import scipy.sparse as sparse
import tensorflow as tf
from scipy.sparse import coo_array, csr_matrix
from sklearn import logger, metrics, preprocessing
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.train import latest_checkpoint

# Directory to store model checkpoints
CHECKPOINT_DIR = os.path.join(
    "data", "models", "mlp_model_v2")
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, "cp_{val_loss:.2f}-{val_accuracy:.2f}.ckpt")
# Path to the history CSV file to store training metrics
HIST_CSV_PATH = os.path.join(CHECKPOINT_DIR, "history.csv")



# Function to convert a sparse matrix into a sparse tensor
def convert_sparse_matrix_to_sparse_tensor(X: csr_matrix) -> tf.SparseTensor:
    """Convert a sparse matrix into a sparse tensor."""
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()  # pyright: ignore
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape)
    )  # pyright: ignore
  
print("Création du modèle de prédiction de catégorie de produit basé sur les mots")
    
# Load data
DATA_FOLDER = os.path.join("datascience", "data")
print(f"Import data from {DATA_FOLDER}")
df = data.load_data(DATA_FOLDER).fillna("")

print("Extract target and features")
features = df["designation"] + " " + df["description"]
target = df["prdtypecode"]


print("Divide dataset into test, train and validation segments")
X_train, X_test, y_train, y_test = train_test_split(
    features, target, random_state=123, test_size=0.2, train_size=0.8
)
X_train.name = "X_train"
X_test.name = "X_test"


print("Transform X_train and X_test")
X_train = preprocessing.transform(X_train)
X_test = preprocessor.transform(X_test)


print("Convert them into sparse matrix")
X_train: sparse.csr_matrix = sparse.csr_matrix(X_train)
X_test: sparse.csr_matrix = sparse.csr_matrix(X_test)


print("Convert target values into a range [0, num_classes - 1]")
y_temp = np.copy(y_train)
for i, c in enumerate(np.unique(y_temp)):
    y_train[y_temp == c] = i

y_temp = np.copy(y_test)
for i, c in enumerate(np.unique(y_temp)):
    y_test[y_temp == c] = i


X_train = convert_sparse_matrix_to_sparse_tensor(X_train)
X_test = convert_sparse_matrix_to_sparse_tensor(X_test)

y_train = y_train.ravel()
y_test = y_test.ravel()


y_test_encode = tf.keras.utils.to_categorical(y_test)
y_train_encode = tf.keras.utils.to_categorical(y_train)


nb_classes = y_train_encode.shape[1]


cp_callbacks = [
    # Stop the training when there is no improvement in val_accuracy for x epochs
    EarlyStopping(monitor="val_accuracy", patience=5),
    # Save a checkpoint
    ModelCheckpoint(CHECKPOINT_PATH,
                    save_best_only=True,
                    mode="max",
                    monitor="val_accuracy",
                    save_weights_only=False,
                    verbose=1),
    # Insert the metrics into a CSV file
    CSVLogger(HIST_CSV_PATH, separator=",", append=True)
]


# Define the model
model = Sequential([
    layers.GlobalAveragePooling2D(),
    layers.Dense(nb_classes, activation="softmax")
])


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train.repeat(),
    y_train_encode,
    epochs=100,
    validation_data=(X_test.repeat(), y_test_encode),
    callbacks=cp_callbacks,
)

# Save the model

output_dir: Path = pydantic.Field(
        description="Directory to save trained model and stats."
    )
logger.info("Save the model")
latest = latest_checkpoint(CHECKPOINT_DIR)
if latest is not None:
    print("Loading checkpoint", latest)
    model.load_weights(latest)
    model.save(args.output_dir / "text_model.keras")
else:
    print("No checkpoint to load")


training_history = pd.read_csv(HIST_CSV_PATH, delimiter=",", header=0)

fig = plt.figure(figsize=(10, 3))
ax1 = fig.add_subplot(121)

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

# Affichage de la figure
plt.show()


plt.savefig(np.argsort.output_dir / "training_text_history.png")

# Predict category on text data
predictions = np.array([])
labels = np.array([])
for x, y in zip(X_test, y_test):
    predictions = np.concatenate(
        [predictions, np.argmax(model.predict(x, verbose=0), axis=-1)]
    )
    labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

prdtypecodes = sorted([int(x) for x in os.listdir(test_dir)])

test_labels = [prdtypecodes[int(i)] for i in labels]
predictions_labels = [prdtypecodes[int(i)] for i in predictions]

# Accuracy score
logger.info(
    f"Accuracy score: {metrics.accuracy_score(test_labels, predictions_labels)}"
)

logger.info("Generate classification report")
class_report = metrics.classification_report(
    test_labels, predictions_labels, zero_division=0.0
)

os.path(args.output_dir / "classification_report.txt").write_text(str(class_report))
logger.info(str(class_report))

# Generate confusion matrix
cnf_matrix = np.round(
    metrics.confusion_matrix(test_labels, predictions_labels, normalize="true"), 2
)

classes = range(0, nb_classes)

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
