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
import pandas as pd
import numpy 

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

# def get_top_features(vectorizer, model, file, n=10):
#     #get the actual word names from the vectorizer
#     feature_names = vectorizer.get_feature_names_out()

#     all_genres_words = {}
#     #loop through 4 classes
#     for i, class_label in enumerate(model.classes_):
#         #sort the log-probabilities for the current class
#         top_indices = np.argsort(model.feature_log_prob_[i])[-n:]

#         #pull the corresponding words
#         top_words = [feature_names[idx] for idx in top_indices]
        

#         print(f"Top words for Class {class_label}:", file=file)
#         print(", ".join(top_words), file=file)
#         print("-" * 30, file=file)


def get_naive_bayes_features(vectorizer, model, file=None, n=10, use_ratio=True):
    feature_names = vectorizer.get_feature_names_out()
    feature_log_probs = model.feature_log_prob_
    feature_dict = {}

    # map numeric class labels to genre names
    genre_names = {1: 'Romance', 2: 'Horror', 3: 'Comedy', 4: 'Action'}

    for i, class_label in enumerate(model.classes_):
        genre = genre_names.get(class_label, str(class_label))  # fallback to str if unexpected

        if use_ratio and len(model.classes_) > 1:
            other_classes = [j for j in range(len(model.classes_)) if j != i]
            mean_other_log_prob = np.mean(feature_log_probs[other_classes], axis=0)
            scores = feature_log_probs[i] - mean_other_log_prob
        else:
            scores = feature_log_probs[i]

        top_indices = np.argsort(scores)[-n:]
        top_words = [feature_names[idx] for idx in reversed(top_indices)]
        top_scores = [scores[idx] for idx in reversed(top_indices)]

        # use genre name as key instead of raw class label
        feature_dict[genre] = dict(zip(top_words, top_scores))

        if file is not None:
            print(f"Top Distinctive Words for {genre}:", file=file)
            print(", ".join(top_words), file=file)
            print("-" * 30, file=file)

    return feature_dict

def get_top_lstm_features(model, X_test, vectorizer, file=None, n=10):
    genre_names = ['Romance', 'Horror', 'Comedy', 'Action']

    if hasattr(vectorizer, 'word_index'):
        index_to_word = {idx: word for word, idx in vectorizer.word_index.items()}
    else:
        vocab = vectorizer.get_vocabulary()
        index_to_word = {idx: word for idx, word in enumerate(vocab)
                         if word not in ('', '[UNK]')}

    pred_proba = model.predict(X_test, verbose=0)
    feature_dict = {}

    for class_idx, genre in enumerate(genre_names):
        class_confidences = pred_proba[:, class_idx]
        top_sample_indices = np.argsort(class_confidences)[-50:]

        word_scores = {}
        for sample_idx in top_sample_indices:
            confidence = class_confidences[sample_idx]
            token_ids = X_test[sample_idx]
            for token_id in token_ids:
                if token_id == 0:
                    continue
                word = index_to_word.get(int(token_id))
                if word:
                    word_scores[word] = word_scores.get(word, 0) + float(confidence)

        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n]

        # use genre name as key instead of class_idx
        feature_dict[genre] = dict(top_words)

        if file is not None:
            print(f"Top Distinctive Words for {genre}:", file=file)
            print(", ".join([w for w, _ in top_words]), file=file)
            print("-" * 30, file=file)

    return feature_dict

def get_top_transformer_features(model, X_test, vectorizer, n=10):
    '''
    Gets top words for each class based on which test samples the
    transformer is most confident about, then finds frequent words in those.
    '''
    genre_names = ['Romance', 'Horror', 'Comedy', 'Action']

    # invert vocab to map index -> word
    if hasattr(vectorizer, 'word_index'):
        index_to_word = {idx: word for word, idx in vectorizer.word_index.items()}
    else:
        vocab = vectorizer.get_vocabulary()
        index_to_word = {idx: word for idx, word in enumerate(vocab)
                         if word not in ('', '[UNK]')}

    # get predicted probabilities for all test samples
    pred_proba = model.predict(X_test, verbose=0)  # (n_samples, 4)

    feature_dict = {}

    for class_idx, genre in enumerate(genre_names):
        # get confidence scores for this class across all samples
        class_confidences = pred_proba[:, class_idx]

        # take the top n most confident samples for this class
        top_sample_indices = np.argsort(class_confidences)[-50:]

        # count word frequencies across those samples
        word_scores = {}
        for sample_idx in top_sample_indices:
            confidence = class_confidences[sample_idx]
            token_ids = X_test[sample_idx]
            for token_id in token_ids:
                if token_id == 0:
                    continue  # skip padding
                word = index_to_word.get(int(token_id))
                if word:
                    # weight by model confidence so high-confidence samples count more
                    word_scores[word] = word_scores.get(word, 0) + float(confidence)

        # sort and take top n
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        feature_dict[genre] = dict(top_words)

    return feature_dict



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