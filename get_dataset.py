import os
from pathlib import Path

import typer
from huggingface_hub import snapshot_download

# Local directory where the dataset will be downloaded
dataset_path = Path("./dataset")
dataset_dir = dataset_path / "WhaleDrone"

# Local directory where the dataset will be downloaded
model_dir = dataset_path / "WhaleDrone_model"


def download_whaledrone_dataset(minimal=False):

    if minimal:
        allow_patterns = [
            "*/DJI_2026011419*",
            "MX_2026_platform_specs.csv",
            "README.md",
            "TECHNICAL_README.md",
            "camera_calibration_DJIM3T_RGBwide.json",
            "dataset_index.xlsx"
        ]
    else:
        allow_patterns = None
    snapshot_download(
        repo_id="LucieLprt-Dvldr/WhaleDrone",
        repo_type="dataset",
        revision="e78c4db",  # for strict reproducibility, use "main" otherwise
        local_dir=dataset_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=[".git*"],  # Prevents downloading internal Git history files
    )

    print(f"WhaleDrone dataset successfully downloaded to: {dataset_dir}")


def download_whaledrone_model():

    print("Starting download of the WhaleDrone model ...")

    snapshot_download(
        repo_id="pierreadorni/WhaleDrone",
        repo_type="model",
        revision="093652a",  # for strict reproducibility, use "main" otherwise
        local_dir=model_dir,
        ignore_patterns=[".git*"],  # Prevents downloading internal Git history files
    )

    print(f"WhaleDrone model successfully downloaded to: {model_dir}")


def main(
    minimal: bool = typer.Option(
        False,
        "--minimal",
        "-m",
        help="Download only a few examples files from the Whale Drone dataset.",
    ),
):
    """
    Utility program to download the WhaleDrone dataset and the corresponding model.
    """
    if minimal:
        typer.echo("Download a minimal sample of the dataset files ")
    else:
        typer.echo("Download the full dataset")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    download_whaledrone_dataset(minimal)
    download_whaledrone_model()


if __name__ == "__main__":
    typer.run(main)
