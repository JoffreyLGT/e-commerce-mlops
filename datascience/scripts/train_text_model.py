"""Script to generate a text preprocessing pickle.

This pickle is used to preprocess the data and send them to the model.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false

import os
import pickle

import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
from scipy.sparse import coo_array, csr_matrix
from sklearn.model_selection import train_test_split

from datascience.src import data, text_model


def convert_sparse_matrix_to_sparse_tensor(X: csr_matrix) -> tf.SparseTensor:
    """Convert a sparse matrix into a sparse tensor."""
    coo: coo_array = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()  # pyright: ignore
    return tf.sparse.reorder(
        tf.SparseTensor(indices, coo.data, coo.shape)
    )  # pyright: ignore


print("Generation text preprocessor")

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

# debug
PICKLE_DEST = os.path.join(DATA_FOLDER, "pickles", "text_preprocessor.pkl")
with open(PICKLE_DEST, "rb") as fp:
    preprocessor = pickle.load(fp)
# end debug

# print("Text preprocessor type: TfidfStemming")
# preprocessor = TextPreprocess(TfidfStemming())
# print("Fit the features into the text preprocessor")
# preprocessor.fit(X_train)

# PICKLE_DEST = os.path.join(DATA_FOLDER, "pickles", "text_preprocessor.pkl")
# print(f"Save text preprocessor into {PICKLE_DEST}")
# with open(PICKLE_DEST, "wb") as fp:
#     pickle.dump(preprocessor, fp)

print("Transform X_train and X_test")
X_train = preprocessor.transform(X_train)
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

dataset = ((X_train, y_train_encode), (X_test, y_test_encode))
nb_classes = y_train_encode.shape[1]
text_model.train_ngram_model(dataset, nb_classes, layers=2, units=512 / 4)  # units=512)
