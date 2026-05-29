'''
process.py

Cleans and tokenizes the data and adds summaries
'''

import re
import os
import pickle
import string
import spacy
import pandas as pd
from string import punctuation
from heapq import nlargest
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS


def remove_actors(text):
  '''removes actor names from text description'''
  # get rid of opening parenthesis, multiple characters that are not 
  # parenthesis, and closing parenthesis
  no_actors = re.sub(r'\([^)]+\)', '', text)

  #only keep letters, lowercase, and get rid of punctuation
  clean_text = re.sub(r'[^a-zA-Z]', ' ', no_actors.lower())

  # Optional: Clean up double spaces left behind
  return clean_text.strip().replace('  ', ' ')



def tokenize(text):
    '''tokenize the given test and remove stopwords'''
    lemmatizer = WordNetLemmatizer() #import lemmatizer

    stop_words = set(stopwords.words('english')).union(set(string.punctuation))
    raw_tokens = word_tokenize(text)

    filtered_tokens = []
    for word in raw_tokens:
        #double check for stop words and empty strings
        if word not in stop_words and len(word) > 1:
            #lemmatize
            lemma = lemmatizer.lemmatize(word)
            filtered_tokens.append(lemma)

    #rejoin to one string
    return ' '.join(filtered_tokens)



def remove_numbers_spacy(text):
    '''remove numbers and number-words from the given text'''
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # Join tokens only if they aren't numbers (1, 2, 3) or number-words (one, two)
    return " ".join([token.text for token in doc if not token.like_num])



def spacy_summarizer(text, limit=2):
    '''summarize the given test'''
    #load the model and process the text
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    #build a frequency table for words
    word_frequencies = {}
    for word in doc:
        #ignore stop words and punctuation
        if word.text.lower() not in STOP_WORDS and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    #normalize frequencies (divide by max frequency)
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    #score the sentences
    sentence_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    #select the top N sentences
    summary = nlargest(limit, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    return " ".join(final_summary)



def process_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    '''processes the text data and produces summaries before saving to pickle'''

    # convert genres to numeric categories
    genre_to_nums = {'romance':1, 'horror':2, 'comedy':3, 'action':4}
    df['genre_map'] = df['genre'].map(genre_to_nums)

    # Clean and tokenize original movie description
    df['Clean_Summary'] = df['description'].apply(remove_actors)
    df['Tokenized_Summary'] = df['Clean_Summary'].apply(tokenize)
    df['Final_Summary'] = df['Tokenized_Summary'].apply(remove_numbers_spacy)

    # create a summary using spacy
    if cfg.get("summary", True):
        df['Summary'] = (df['description'].apply(lambda x: spacy_summarizer(x, limit=2)))
    
    # save data to pickle file
    files = cfg.get("files", {})
    paths = cfg.get("paths", {})

    os.makedirs(paths['processed_data'], exist_ok=True)

    with open(files['data_file'], "wb") as f:
            pickle.dump(df, f)

    return df



