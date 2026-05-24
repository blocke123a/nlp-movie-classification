'''
utils.py

Helper utility functions
'''
import yaml
import numpy as np
from pathlib import Path
from typing import Union

def load_config(path: Union[str, Path] = "config.yaml") -> dict:
    '''loads given config file'''
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    try:
        with open(path, "r", encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            if cfg is None:
                print(f'''[warning] Config file {path.resolve()} is empty.
                       Returning empty dict.''')
                return {}
            return cfg
    except yaml.YAMLError as ye:
        raise yaml.YAMLError(f"Error parsing YAML file {path.resolve()}: {ye}")
    except Exception as e:
        raise RuntimeError(f'''Unexpected error loading config
                            {path.resolve()}: {e}''') from e

def get_top_features(vectorizer, model, file, n=10):
    #get the actual word names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    #loop through 4 classes
    for i, class_label in enumerate(model.classes_):
        #sort the log-probabilities for the current class
        top_indices = np.argsort(model.feature_log_prob_[i])[-n:]

        #pull the corresponding words
        top_words = [feature_names[idx] for idx in top_indices]

        print(f"Top words for Class {class_label}:", file=file)
        print(", ".join(top_words), file=file)
        print("-" * 30, file=file)