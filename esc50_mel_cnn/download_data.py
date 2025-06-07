import os
from pathlib import Path

# TODO use fire

# TODO remove global variables
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/mmoreaux/"
    + "environmental-sound-classification-50"
)


def download_esc50(dataset_path: str | Path) -> None:
    """Download ESC-50 dataset if it is not already downloaded.

    Args:
        dataset_path (str | Path): path to the root direrectory of the dataset.
    """
    dataset_path = Path(dataset_path)
    if dataset_path.exists() and (dataset_path / "esc50.csv").exists():
        return

    print("Downloading ESC-50 dataset...")
    dataset_path.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_path / "tmp_data.zip"

    # TODO use requests?
    download_command = f"curl -L -o {zip_path} {URL}"
    print(f"Executing: {download_command}")
    os.system(download_command)

    # TODO use shutil?
    unzip_command = f"unzip -q {zip_path} -d {dataset_path}"
    print(f"Executing: {unzip_command}")
    os.system(unzip_command)

    print(f"Deleting {zip_path}...")
    zip_path.unlink()
    print("ESC-50 dataset is ready.")
