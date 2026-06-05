# update_data.py – runs the full pipeline and overwrites model & db
import subprocess
import sys

def run_pipeline():
    print("1. Downloading latest data...")
    subprocess.run([sys.executable, "src/ingestion.py"], check=True)

    print("2. Running preprocessing...")
    subprocess.run([sys.executable, "src/preprocessing.py"], check=True)

    print("3. Training model...")
    subprocess.run([sys.executable, "src/train.py"], check=True)

    print("Update complete. New model and database saved.")
    
if __name__ == "__main__":
    run_pipeline()