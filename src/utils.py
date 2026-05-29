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
    """
    Generalized feature extraction for naive bayes model
    """
    feature_names = vectorizer.get_feature_names_out()
    feature_log_probs = model.feature_log_prob_
    feature_dict = {}
    
    for i, class_label in enumerate(model.classes_):
        if use_ratio and len(model.classes_) > 1:
            #calculate the average log probability of all other classes
            other_classes = [j for j in range(len(model.classes_)) if j != i]
            mean_other_log_prob = np.mean(feature_log_probs[other_classes], axis=0)
            
            #ratio shows how unique the word is to this class specifically
            scores = feature_log_probs[i] - mean_other_log_prob
        else:
            #fallback to raw log probabilities
            scores = feature_log_probs[i]
            
        #get the indices of the top n scores
        top_indices = np.argsort(scores)[-n:]
        
        #sort from highest impact down to lowest
        top_words = [feature_names[idx] for idx in reversed(top_indices)]
        top_scores = [scores[idx] for idx in reversed(top_indices)]
        
        #save to dictionary
        feature_dict[class_label] = dict(zip(top_words, top_scores))
        
        #output to console
        if file is not None:
            print(f"Top Distinctive Words for Class {class_label}:", file=file)
            print(", ".join(top_words), file=file)
            print("-" * 30, file=file)
            
    return feature_dict

def get_top_lstm_features(model, vectorizer, file=None, n = 10):
    embedding_weights = model.layers[1].get_weights()[0]
    dense_weights = model.layers[-1].get_weights()[0]
    
    #class space by matrix multiplying
    word_class_scores = np.dot(embedding_weights, dense_weights)

    if hasattr(vectorizer, 'word_index'):
        vocab = vectorizer.word_index
    else:
        vocab = vectorizer.vocabulary_ #fallback

    #invert the dictionary to map index to word
    index_to_word = {idx: word for word, idx in vocab.items()}

    feature_dict = {}

    num_classes = dense_weights.shape[1]
    for class_idx in range(num_classes):
        class_scores = word_class_scores[:,class_idx]

        df_list = []
        for idx, score in enumerate(class_scores):
            word = index_to_word.get(idx,None)
            if word: #skip padding
                df_list.append({'feature': word, 'coefficient': score})
        #make into dataframe
        class_df = pd.DataFrame(df_list)

        top_features = class_df.sort_values(by='coefficient', ascending=False).head(n)

        feature_dict[class_idx] = dict(zip(top_features['feature'], top_features['coefficient']))

        print(f"Top {n} LSTM Features for Class {class_idx}:")
        print(top_features.to_string(index=False))
        print("-" * 30)

    return feature_dict

def get_top_transformer_features(model, X_test, vectorizer, n=10):
    '''
    Gets top words for each class based on transformer attention weights
    '''

    #get attention layer
    attn_layer = None
    for layer in model.layers:
        if 'attention' in layer.name.lower():
            attn_layer = layer
    if attn_layer is None:
        raise ValueError("Could not find an attention layer in the model")
    
    #get embedding weights
    embedding_layer = [l for l in model.layers if 'embed' in l.name.lower()][0]
    dense_layer = model.layers[-1]

    embed_weights = embedding_layer.get_weights()[0] if embedding_layer.get_weights() else model.layers[1].get_weights()[0]
    dense_weights = dense_layer.get_weights()[0]

    word_class_scores = np.dot(embed_weights, dense_weights) #get dot product

    #format vocab matching
    vocab = vectorizer.get_vocabulary() if hasattr(vectorizer, 'get_vocabulary') else list(vectorizer.word_index.keys())

    #create feature dict
    feature_dict = {}
    genre_names = ['Romance', 'Horror','Comedy','Action']
    for class_idx in range(dense_weights.shape[1]):
        class_scores = word_class_scores[:, class_idx]
        
        df_list = []
        for idx, score in enumerate(class_scores):
            if idx < len(vocab):
                word = vocab[idx]
                if word not in ['', '[UNK]']:
                    df_list.append({'feature': word, 'coefficient': score})
                    
        class_df = pd.DataFrame(df_list)
        
        #sort to find the highest positive directional weights
        top_features = class_df.sort_values(by='coefficient', ascending=False).head(n)
        
        #map the structural index - numbers to genre names
        genre_string = genre_names[class_idx]
        
        #convert to frequency dict format
        feature_dict[genre_string] = dict(zip(top_features['feature'], top_features['coefficient']))
        
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