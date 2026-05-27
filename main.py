'''
main.py

'''

import logging
import os
import pandas as pd
from pathlib import Path
from src import load_config, get_data, process_data
from src import train_bayes, train_log_reg, train_lstm, train_transformer

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
    
    if models['logistic_regression']:
        train_log_reg(df, cfg)

    if models['transformer']:
        train_transformer(df, cfg)

    if models['lstm']:
        train_lstm(df, cfg)


if __name__=="__main__":
    main()