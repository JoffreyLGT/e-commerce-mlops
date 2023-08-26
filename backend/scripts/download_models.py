"""Download the models and preprocessing saves from Google Drive."""

from pathlib import Path

import gdown

from app.core.settings import backend_dir

files = [
    (
        "https://drive.google.com/uc?id=1BeeAydZtfeqW1PIoM2O6SuD7lSKOr84S",
        Path(backend_dir) / "datascience" / "data" / "models" / "text",
        "mlp_model_v2.h5",
    ),
    (
        "https://drive.google.com/uc?id=1eCW7UZ6oKyLvt_cn5Ek4vzNa55YZAU70",
        Path(backend_dir) / "datascience" / "data" / "models" / "image",
        "cnn_mobilenetv2.h5",
    ),
    (
        "https://drive.google.com/uc?id=1CwMWDs6Sb7EDNoOM4tzC5Eyo2DOP2G8M",
        Path(backend_dir) / "datascience" / "data" / "models" / "fusion",
        "fusion_text_image.h5",
    ),
    (
        "https://drive.google.com/uc?id=1koGV4Zk0gWDeUSedrQFJsIZZTQsByV78",
        Path(backend_dir) / "datascience" / "data" / "pickles",
        "text_preprocessor.pkl",
    ),
]


def main() -> None:
    """Main function triggered only when the script is called."""
    for url, destination, filename in files:
        if not Path.exists(destination):
            Path(destination).mkdir(parents=True)
        full_path = Path(destination) / filename
        if not Path.is_file(full_path):
            gdown.download(url, str(full_path), quiet=False)


# Safety net to call main only when the script is called by Python interpreter
if __name__ == "__main__":
    main()
