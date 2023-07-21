"""
Download the models and preprocessing saves from Google Drive.
"""

import os

import gdown

files = [
    (
        "https://drive.google.com/uc?id=1BeeAydZtfeqW1PIoM2O6SuD7lSKOr84S",
        os.path.join("datascience", "data", "models", "text"),
        "mlp_model_v2.h5",
    ),
    (
        "https://drive.google.com/uc?id=1eCW7UZ6oKyLvt_cn5Ek4vzNa55YZAU70",
        os.path.join("datascience", "data", "models", "image"),
        "cnn_mobilenetv2.h5",
    ),
    (
        "https://drive.google.com/uc?id=1CwMWDs6Sb7EDNoOM4tzC5Eyo2DOP2G8M",
        os.path.join("datascience", "data", "models", "fusion"),
        "fusion_text_image.h5",
    ),
    (
        "https://drive.google.com/uc?id=1koGV4Zk0gWDeUSedrQFJsIZZTQsByV78",
        os.path.join("datascience", "data", "pickles"),
        "text_preprocessor.pkl",
    ),
]

for url, destination, filename in files:
    if not os.path.exists(destination):
        os.makedirs(destination)
    full_path = os.path.join(destination, filename)
    if not os.path.isfile(full_path):
        gdown.download(url, full_path, quiet=False)
