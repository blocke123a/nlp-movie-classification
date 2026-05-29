'''
naive_bayes.py

Train and pickle a naive bayes model on the given dataframe
'''
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from .utils import get_naive_bayes_features
from wordcloud import WordCloud

def train_bayes(df: pd.DataFrame, cfg: dict):
    '''train and save a naive bayes model'''

    params = cfg.get("model_parameters", {})
    paths = cfg.get("paths", {})
    # get model paths
    nb_model_dir = paths['models']
    nb_metrics_dir = paths['metrics'] + "/naive_bayes"
    model_file = nb_model_dir + "/naive_bayes.joblib"

    # make directories
    os.makedirs(paths['models'], exist_ok=True)
    os.makedirs(paths['metrics'], exist_ok=True)
    os.makedirs(nb_model_dir, exist_ok=True)
    os.makedirs(nb_metrics_dir, exist_ok=True)


    X = df['Final_Summary'] # create X
    y = df['genre_map'] # create y

    # make train/test split
    X_train, X_test, y_train, y_test = (train_test_split(X, y, 
                                        test_size=params['test_split'],
                                        random_state=params['random_state']))
        
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # make train document term matrix
    X_train_dtm = vectorizer.fit_transform(X_train)

    # make test document term matrix
    X_test_dtm = vectorizer.transform(X_test)
    
    # train model if it doesn't exist already
    if not os.path.exists(model_file):
        model = MultinomialNB() #define multinomial bayes model
        model.fit(X_train_dtm, y_train) #fit on train document term matrix and y_train

        joblib.dump(model, model_file)
        joblib.dump(vectorizer, nb_model_dir + 'tfidf_vectorizer.joblib')
    else:
        # load existing model
        model = joblib.load(model_file)

    # evaluate the model
    evaluate_bayes(model, X_test_dtm, y_test, paths, vectorizer)



def evaluate_bayes(model, X_test, y_test, paths, vectorizer):
    '''evaluates the naive bayes model'''
    nb_metrics_dir = paths['metrics'] + "/naive_bayes"

    preds = model.predict_proba(X_test) #predict on test set

    preds /= preds.sum(axis=1)[..., np.newaxis] #normalize so they are all 0 to 1

    y_pred = model.predict(X_test)

    evaluation_file = nb_metrics_dir + "/naive_bayes_evaulation.txt"
    cm_file = nb_metrics_dir + "/naive_bayes_cm.jpg"
    wordcloud_file = nb_metrics_dir + "/naive_bayes_wordcloud.jpg"

    with open(evaluation_file, "w", encoding='utf-8') as f:
        print("====== Model Evaluation: Naive Bayes ======\n", file=f)

        print("=== Model Score ===", file=f)
        print(f'{model.score(X_test, y_test)=}', file=f)
        print("\n", file=f)

        #classification report
        print("=== Classification Report ===", file=f)
        print(classification_report(y_test, y_pred), file=f)
        print("\n", file=f)

        print("=== Accuracy Score ===", file=f)
        print(f'{accuracy_score(y_test, y_pred)=}', file=f) #get accuracy

        print("\n=== Top Features for the Naive Bayes model ===\n", file=f)
        genre_word_frequencies = get_naive_bayes_features(vectorizer, model, f)
        get_naive_bayes_features(vectorizer, model, f)

    cm = confusion_matrix(y_test, y_pred)

    disp = (ConfusionMatrixDisplay(confusion_matrix=cm, 
                    display_labels=['romance', 'horror','comedy','action']))
    disp.plot(cmap=plt.cm.Greens)
    plt.title('Naive Bayes Confusion Matrix')
    plt.savefig(cm_file)

    #wordcloud
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten() #flatten 2D array to 1D

    for idx, (genre, frequencies) in enumerate(genre_word_frequencies.items()):
        #define a mapping of genres to specific color schemes
        color_map_selection = {
            'Romance': 'RdPu',
            'Horror': 'magma',       # Dark purple/orange vibe
            'Comedy': 'spring',      # Bright and energetic
            'Action': 'viridis'      # Bold contrast
        }
        
        #fallback
        current_cmap = color_map_selection.get(genre, 'viridis')

        #pass the colormap to WordCloud
        wc = WordCloud(
            background_color='white', 
            width=800, 
            height=400, 
            max_words=40,
            colormap=current_cmap
        )
        # Generate using the coefficients as weights
        wc.generate_from_frequencies(frequencies)
        
        #plot on subplots
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'Top Words for {genre}', fontsize=16)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(wordcloud_file, dpi=300)
    plt.close()