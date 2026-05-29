'''
data-ingestion.py

Ingests and returns the data from the config kaggle link
'''
import pandas as pd
import os
import subprocess
from pathlib import Path

def get_data(cfg: dict) -> pd.DataFrame:
    '''gets the data from the provided kaggle link'''

    links = cfg.get("data", {})
    kaggle_link = links['kaggle_link']
    data_dir = "data"

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    csv_file = Path(data_dir) / "IMDB_four_genre_larger_plot_description.csv"

    # Download if not already present
    if not csv_file.exists():
        print(f"Downloading dataset: {kaggle_link}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_link, "-p", data_dir],
            check=True
        )
        # Unzip if needed
        import zipfile
        zip_file = Path(data_dir) / f"{kaggle_link.split('/')[-1]}.zip"
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_file.unlink()

    df = pd.read_csv(csv_file)
    return df