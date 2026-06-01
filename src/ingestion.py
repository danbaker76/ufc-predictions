import kagglehub
import os
import shutil

RAW_DIR = "data/raw"

def download_data():
    path = kagglehub.dataset_download("rajeevw/ufcdata")
    os.makedirs(RAW_DIR, exist_ok=True)
    for fname in os.listdir(path):
        shutil.copy(os.path.join(path, fname), RAW_DIR)
    print("Data downloaded to", RAW_DIR)

if __name__ == '__main__':
    download_data()