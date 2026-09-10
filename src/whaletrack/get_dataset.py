import os
from pathlib import Path

from huggingface_hub import snapshot_download

# Local directory where the dataset will be downloaded
dataset_path = Path("./dataset")
dataset_dir = dataset_path / "WhaleDrone"

# Local directory where the dataset will be downloaded
model_dir = dataset_path / "WhaleDrone_model"

def download_whaledrone_dataset():
    
    
    print("Starting download of the WhaleDrone dataset (135 GB)...")
    
    snapshot_download(
        repo_id="LucieLprt-Dvldr/WhaleDrone",
        repo_type="dataset",
        revision="e78c4db",  # for strict reproducibility, use "main" otherwise
        local_dir=dataset_dir,
        local_dir_use_symlinks=False, # Physically downloads MP4 files into the local folder
        ignore_patterns=[".git*"],    # Prevents downloading internal Git history files
        resume_download=True          # Resumes interrupted downloads automatically
    )

    print(f"WhaleDrone dataset successfully downloaded to: {dataset_dir}")


def download_whaledrone_model():
    
    
    print("Starting download of the WhaleDrone model ...")
    
    snapshot_download(
        repo_id="pierreadorni/WhaleDrone",
        repo_type="model",
        revision="093652a", # for strict reproducibility, use "main" otherwise
        local_dir=model_dir,
        local_dir_use_symlinks=False, # Physically downloads MP4 files into the local folder
        ignore_patterns=[".git*"],    # Prevents downloading internal Git history files
        resume_download=True          # Resumes interrupted downloads automatically
    )

    print(f"WhaleDrone model successfully downloaded to: {model_dir}")





if __name__ == "__main__":
    print(os.getcwd())

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    download_whaledrone_dataset()
    download_whaledrone_model()
