'''
utils.py

Helper utility functions
'''
import yaml
import os
import pickle
import requests
import zipfile
import numpy as np
from pathlib import Path
from typing import Union
from tensorflow.keras.layers import Embedding

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


def get_embeddings_from_file(file_name):
    '''gets embeddings from provided file'''
    embeddings_index = {}
    with open(file_name, encoding="utf8") as f:
        for line in f:
            values = line.rstrip().split(' ')
            if len(values) > 2:
                embeddings_index[values[0]] = np.asarray(values[1:], dtype="float32")
    return embeddings_index


def get_embeddings():
    '''Get FastText embeddings'''

    # URL and paths
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
    zip_name = "wiki-news-300d-1M.vec.zip"
    file_name = "wiki-news-300d-1M.vec"

    # download zip file if doesn't exist
    if not os.path.exists(zip_name) and not os.path.exists(file_name):
        response = requests.get(url, stream=True)
        with open(zip_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # unzip file
    if os.path.exists(zip_name) and not os.path.exists(file_name):
        print("Unzipping file")
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_name)
        print("File ready")

    # get embeddings from file
    return get_embeddings_from_file(file_name)


def get_embedding_matrix(word_index,embeddings_index, embeddings_dim=300):
    '''make embedding matrix'''
    nb_words = len(word_index) + 1 # +1 since min(word_index.values())=1
    embedding_matrix = np.zeros((nb_words,embeddings_dim))
    unknown = 0

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None: unknown += 1
        else: embedding_matrix[i] = embedding_vector

    return embedding_matrix, unknown


def make_save_emb_layer(word_index, embeddings_index, layer_file_name):
    '''make and save the embedding layer'''
    embedding_matrix, unknown = get_embedding_matrix(word_index, embeddings_index)
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)
    with open(layer_file_name, 'wb') as f:
        pickle.dump(embedding_layer, f, -1)
    return unknown