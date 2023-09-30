"""Download the default mlruns.zip and unzip it in deployment directory."""

import zipfile
from pathlib import Path

import gdown


def main() -> None:
    """Main function triggered only when the script is called."""
    destination = Path("deployment")
    filename = "mlruns.zip"
    url = "https://drive.google.com/uc?id=1vayh57QSoGHaKm0wsOr1VL_85fIK88LQ"
    destination.mkdir(parents=True, exist_ok=True)
    full_path = destination / filename
    if not Path.is_file(full_path):
        gdown.download(url, str(full_path), quiet=False)

    with zipfile.ZipFile(full_path, "r") as zip_ref:
        zip_ref.extractall(destination)

    full_path.unlink()


# Safety net to call main only when the script is called by Python interpreter
if __name__ == "__main__":
    main()
