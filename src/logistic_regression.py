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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from wordcloud import WordCloud

def train_log_reg(df: pd.DataFrame, cfg: dict):
    '''train and save a naive bayes model'''

    params = cfg.get("model_parameters", {})
    paths = cfg.get("paths", {})
    # get model paths
    lr_model_dir = paths['models']
    lr_metrics_dir = paths['metrics'] + "/logistic_regression"
    model_file = lr_model_dir + "/logistic.joblib"

    # make directories
    os.makedirs(paths['models'], exist_ok=True)
    os.makedirs(paths['metrics'], exist_ok=True)
    os.makedirs(lr_model_dir, exist_ok=True)
    os.makedirs(lr_metrics_dir, exist_ok=True)


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
        # set up logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_dtm, y_train) #fit on train document term matrix and y_train

        joblib.dump(model, model_file)
        joblib.dump(vectorizer, lr_model_dir + 'tfidf_vectorizer.joblib')
    else:
        # load existing model
        model = joblib.load(model_file)

    # evaluate the model
    evaluate_log_reg(model, X_test_dtm, y_test, paths, vectorizer)



def get_logreg_features(vectorizer, model, file, n=10):
    words = vectorizer.get_feature_names_out()
    class_dict = {1: "Romance", 2: "Horror", 3: "Comedy", 4: "Action"}

    all_genres_words = {}

    # model.classes_ contains 4 labels
    for i, class_name in enumerate(model.classes_):
        # get coefficients for this specific class
        coeffs = model.coef_[i]

        #create a temp DataFrame for this class
        df = pd.DataFrame({'Word': words, 'Coefficient': coeffs})

        #sort by coefficient (Highest = Most Positive Impact)
        top_features = df.sort_values(by='Coefficient', ascending=False).head(n)

        #put top words in dictionary
        word_freq = dict(zip(top_features['Word'], top_features['Coefficient']))
        all_genres_words[class_dict[class_name]] = word_freq

        print(f"Top 10 Words for Class: {class_dict[class_name]}", file=file)
        print(top_features.to_string(index=False), file=file)
        print("-" * 30, file=file)
    return all_genres_words



def evaluate_log_reg(model, X_test, y_test, paths, vectorizer):
    '''evaluates the logistic regression model'''
    lr_metrics_dir = paths['metrics'] + "/logistic_regression"

    preds = model.predict_proba(X_test) #predict on test set

    preds /= preds.sum(axis=1)[..., np.newaxis] #normalize so they are all 0 to 1

    y_pred = model.predict(X_test)

    evaluation_file = lr_metrics_dir + "/logistic_regression_evaulation.txt"
    cm_file = lr_metrics_dir + "/logistic_regression_cm.jpg"
    wordcloud_file = lr_metrics_dir + "/logistic_regression_wordcloud.jpg"

    with open(evaluation_file, "w", encoding='utf-8') as f:
        print("====== Model Evaluation: Logistic Regression ======\n", file=f)

        print("=== Model Score ===", file=f)
        print(f'{model.score(X_test, y_test)=}', file=f)
        print("\n", file=f)

        #classification report
        print("=== Classification Report ===", file=f)
        print(classification_report(y_test, y_pred), file=f)
        print("\n", file=f)

        print("=== Accuracy Score ===", file=f)
        print(f'{accuracy_score(y_test, y_pred)=}', file=f) #get accuracy

        print("\n=== Top Features for the Logistic Regression model ===\n", file=f)
        genre_word_frequencies = get_logreg_features(vectorizer, model, f)
        get_logreg_features(vectorizer, model, f)
    
    #confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = (ConfusionMatrixDisplay(confusion_matrix=cm, 
                    display_labels=['romance', 'horror','comedy','action']))
    disp.plot(cmap=plt.cm.Purples)
    plt.title('Logistic Regression Confusion Matrix')
    plt.savefig(cm_file)

    #wordcloud
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten() #flatten 2D array to 1D

    for idx, (genre, frequencies) in enumerate(genre_word_frequencies.items()):
        #create wordcloud
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