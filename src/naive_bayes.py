'''
naive_bayes.py

Train and pickle a naive bayes model on the given dataframe
'''
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from .utils import get_top_features

def train_bayes(df: pd.DataFrame, cfg: dict):
    '''train and save a naive bayes model'''

    params = cfg.get("model_parameters", {})
    paths = cfg.get("paths", {})
    nb_model_dir = paths['models'] + "/naive_bayes"
    nb_metrics_dir = paths['metrics'] + "/naive_bayes"

    os.makedirs(paths['models'], exist_ok=True)
    os.makedirs(paths['metrics'], exist_ok=True)
    os.makedirs(nb_model_dir, exist_ok=True)
    os.makedirs(nb_metrics_dir, exist_ok=True)

    X = df['Final_Summary'] #create X
    y = df['genre_map'] #create y

    # make train/test split
    X_train, X_test, y_train, y_test = (train_test_split(X, y, 
                                        test_size=params['test_split'],
                                         random_state=params['random_state']))
    
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    #make train document term matrix
    X_train_dtm = vectorizer.fit_transform(X_train)

    #make test document term matrix
    X_test_dtm = vectorizer.transform(X_test)

    model = MultinomialNB() #define multinomial bayes model
    model.fit(X_train_dtm, y_train) #fit on train document term matrix and y_train

    preds = model.predict_proba(X_test_dtm) #predict on test set

    preds /= preds.sum(axis=1)[..., np.newaxis] #normalize so they are all 0 to 1

    y_pred = model.predict(X_test_dtm)

    evaluation_file = nb_metrics_dir + "/naive_bayes_evaulation.txt"
    cm_file = nb_metrics_dir + "/naive_bayes_cm.jpg"
    model_file = nb_model_dir + "/naive_bayes.pkl"

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    with open(evaluation_file, "w", encoding='utf-8') as f:
        print("====== Model Evaluation: Naive Bayes ======\n", file=f)

        print("=== Model Score ===", file=f)
        print(f'{model.score(X_test_dtm, y_test)=}', file=f)
        print("\n", file=f)

        #classification report
        print("=== Classification Report ===", file=f)
        print(classification_report(y_test, y_pred), file=f)
        print("\n", file=f)

        print("=== Accuracy Score ===", file=f)
        print(f'{accuracy_score(y_test, y_pred)=}', file=f) #get accuracy

        print("\n=== Top Features for the Naive Bayes model ===\n", file=f)
        get_top_features(vectorizer, model, f)

    cm = confusion_matrix(y_test, y_pred)

    disp = (ConfusionMatrixDisplay(confusion_matrix=cm, 
                    display_labels=['romance', 'horror','comedy','action']))
    disp.plot(cmap=plt.cm.Greens)
    plt.title('Confusion Matrix')
    plt.savefig(cm_file)

