import os
import urllib.request
import tarfile

# Define paths
root_path = "./"
ust_data_path = os.path.join(root_path, "data/ust-data/sonyc-ust")

# Create folders
os.makedirs(ust_data_path, exist_ok=True)
os.makedirs(os.path.join(ust_data_path, 'audio-dev'), exist_ok=True)

# File URLs and names
files_to_download = {
    "annotations.csv": "https://zenodo.org/record/3338310/files/annotations.csv",
    "audio-dev.tar.gz": "https://zenodo.org/record/3338310/files/audio-dev.tar.gz",
    "audio-eval.tar.gz": "https://zenodo.org/record/3338310/files/audio-eval.tar.gz",
    "dcase-ust-taxonomy.yaml": "https://zenodo.org/record/3338310/files/dcase-ust-taxonomy.yaml",
    "README.md": "https://zenodo.org/record/3338310/files/README.md"
}

# Download files
for filename, url in files_to_download.items():
    filepath = os.path.join(ust_data_path, filename)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filepath)

# Extract audio-dev.tar.gz
audio_dev_tar_path = os.path.join(ust_data_path, "audio-dev.tar.gz")
audio_eval_tar_path = os.path.join(ust_data_path, "audio-eval.tar.gz")

def extract_tar_file(tar_path, extract_to):
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

extract_tar_file(audio_dev_tar_path, os.path.join(ust_data_path, "audio-dev"))
os.remove(audio_dev_tar_path)

# Extract audio-eval.tar.gz
extract_tar_file(audio_eval_tar_path, ust_data_path)
os.remove(audio_eval_tar_path)

print("All files downloaded and extracted successfully.")


