'''
data-ingestion.py

Ingests and returns the data from the config kaggle link
'''
import pandas as pd
import kagglehub

def get_data(cfg: dict) -> pd.DataFrame:
    '''gets the data from the provided kaggle link'''

    links = cfg.get("data", {})

    data_path = kagglehub.dataset_download(links['kaggle_link'])
    print(data_path)

    df = pd.read_csv(data_path + "/IMDB_four_genre_larger_plot_description.csv")

    return df