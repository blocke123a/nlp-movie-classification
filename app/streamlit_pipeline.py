import numpy as np
import os
import re
import string
import spacy
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from joblib import load
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

GENRES = ['Romance', 'Horror', 'Comedy', 'Action']

# load spacy once at module level since it is expensive to reload per call
nlp = spacy.load("en_core_web_sm")


def remove_actors(text: str) -> str:
    no_actors = re.sub(r'\([^)]+\)', '', text)
    clean_text = re.sub(r'[^a-zA-Z]', ' ', no_actors.lower())
    return clean_text.strip().replace('  ', ' ')


def tokenize(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')).union(set(string.punctuation))
    raw_tokens = word_tokenize(text)
    filtered = []
    for word in raw_tokens:
        if word not in stop_words and len(word) > 1:
            filtered.append(lemmatizer.lemmatize(word))
    return ' '.join(filtered)


def remove_numbers_spacy(text: str) -> str:
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.like_num])


def clean_text(text: str) -> str:
    """Full cleaning pipeline: remove actors → tokenize → remove numbers."""
    text = remove_actors(text)
    text = tokenize(text)
    text = remove_numbers_spacy(text)
    return text

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'max_len': self.max_len, 'd_model': self.d_model})
        return cfg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
MODELS_DIR = os.path.join(BASE_DIR, 'artifacts', 'models')

def load_models():
    models = {
        'naive_bayes': load(os.path.join(MODELS_DIR, 'naive_bayes.joblib')),
        'logistic': load(os.path.join(MODELS_DIR, 'logistic.joblib')),
        'lstm': load_model(os.path.join(MODELS_DIR, 'lstm_genre_model.keras')),
        'transformer': load_model(os.path.join(MODELS_DIR, 'transformer_genre_model.keras'),
                                  custom_objects={'PositionalEncoding': PositionalEncoding})
    }
    with open(os.path.join(MODELS_DIR, 'keras_tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)

    preprocessors = {
        'vectorizer': load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')),
        'tokenizer': tokenizer,
        'max_seq_len': 50,
    }
    return models, preprocessors


def predict(text: str, model_name: str, models: dict, preprocessors: dict) -> dict:
    cleaned = clean_text(text)
    model = models[model_name]

    # stub: random probabilities until models are wired up
    if model is None:
        probs = np.random.dirichlet(np.ones(4))
        return {
            'genre': GENRES[int(np.argmax(probs))],
            'probabilities': dict(zip(GENRES, probs.tolist())),
            'cleaned_text': cleaned,
        }

    # sklearn models
    if model_name in ('naive_bayes', 'logistic'):
        X = preprocessors['vectorizer'].transform([cleaned])
        probs = model.predict_proba(X)[0]

    # keras models
    else:
        seq = preprocessors['tokenizer'].texts_to_sequences([cleaned])
        X = pad_sequences(seq, maxlen=preprocessors['max_seq_len'],
                              padding='post', truncating='post')
        probs = model.predict(X, verbose=0)[0]

    return {
        'genre': GENRES[int(np.argmax(probs))],
        'probabilities': dict(zip(GENRES, probs.tolist())),
        'cleaned_text': cleaned,
    }