'''
main.py

'''

import logging
import os
import pandas as pd
from pathlib import Path
from src import load_config
from src import get_data
from src import process_data
from src import train_bayes

def main(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    data = cfg.get("files", {})
    models = cfg.get("models", {})

    if not os.path.exists(data['data_file']):
        df = get_data(cfg)

        df = process_data(df, cfg)
    else:
        df = pd.read_pickle(data['data_file'])
    
    if models['naive_bayes']:
        train_bayes(df, cfg)


if __name__=="__main__":
    main()