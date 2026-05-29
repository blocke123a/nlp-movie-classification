'''
lstm.py

Train and pickle an LSTM model on the given dataframe
'''
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from .utils import make_save_emb_layer, get_embeddings, get_top_lstm_features
from wordcloud import WordCloud


def get_model(embedding_layer, params):
    '''defines the LSTM in Keras'''
    MAX_SEQUENCE_LENGTH = params['MAX_SEQUENCE_LENGTH']
    NUM_CLASSES = params['NUM_CLASSES']
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_layer)
    x = SpatialDropout1D(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Conv1D(32, kernel_size=3, padding="valid", activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    return model


def train_lstm(df: pd.DataFrame, cfg: dict):
    '''train and save an LSTM'''

    params = cfg.get("model_parameters", {})
    paths = cfg.get("paths", {})

    lstm_model_dir = paths['models']
    lstm_metrics_dir = paths['metrics'] + "/lstm"
    model_file = lstm_model_dir + "/lstm_genre_model.keras"
    weights_file = lstm_model_dir + "/best_lstm_model.weights.h5"
    tokenizer_file = lstm_model_dir + "/keras_tokenizer.pkl"
    layer_file = lstm_model_dir + "embedding_layer.pkl"

    os.makedirs(paths['models'], exist_ok=True)
    os.makedirs(paths['metrics'], exist_ok=True)
    os.makedirs(lstm_model_dir, exist_ok=True)
    os.makedirs(lstm_metrics_dir, exist_ok=True)

    X = df['Final_Summary']
    y = df['genre_map']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                        test_size=params['test_split'],
                                        random_state=params['random_state'])

    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(X_train)

    with open(tokenizer_file, 'wb') as f:
        pickle.dump(tokenizer, f)

    word_index = tokenizer.word_index

    MAX_SEQUENCE_LENGTH = params['MAX_SEQUENCE_LENGTH']

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_lstm = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post',
                                  truncating='post')
    X_test_lstm = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post',
                                 truncating='post')

    # fixed indentation — these must be outside the if block
    y_train = y_train - 1
    y_test = y_test - 1

    if not os.path.exists(model_file):
        embeddings_index = get_embeddings()

        unknown = make_save_emb_layer(word_index, embeddings_index, layer_file)

        with open(layer_file, 'rb') as f:
            embedding_layer = pickle.load(f)

        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(weights_file,
                                           save_best_only=True, save_weights_only=True,
                                           monitor='val_accuracy', mode='max')

        # fixed: pass embedding_layer into get_model
        lstm_model = get_model(embedding_layer, params)

        hist_lstm = lstm_model.fit(
            X_train_lstm, y_train,
            validation_data=(X_test_lstm, y_test),
            epochs=50, batch_size=params['BATCH_SIZE'], shuffle=True, verbose=2,
            callbacks=[model_checkpoint, early_stopping]
        )
        lstm_model.load_weights(weights_file)
        lstm_model.save(model_file)
        evaluate_lstm_training(hist_lstm, paths)

        # fixed: use lstm_model consistently
        model = lstm_model

    else:
        model = load_model(model_file)

    # evaluate the model
    evaluate_lstm(model, tokenizer, X_test_lstm, y_test, paths, params)


def evaluate_lstm(model, vectorizer, X_test, y_test, paths, params):
    '''evaluates the LSTM model'''
    lstm_metrics_dir = paths['metrics'] + "/lstm"

    test_pred_proba = model.predict(X_test, batch_size=params['BATCH_SIZE'], verbose=0)
    test_pred_class = np.argmax(test_pred_proba, axis=1)

    genre_names = ['romance', 'horror', 'comedy', 'action']

    evaluation_file = lstm_metrics_dir + "/lstm_evaulation.txt"
    cm_file = lstm_metrics_dir + "/lstm_cm.jpg"
    arch_file = lstm_metrics_dir + "/lstm_model_architecture.png"
    wordcloud_file = lstm_metrics_dir + "/lstm_model_wordcloud.jpg"

    with open(evaluation_file, "w", encoding='utf-8') as f:
        print("====== Model Evaluation: LSTM ======\n", file=f)

        print("=== Classification Report ===", file=f)
        print(classification_report(y_test, test_pred_class, target_names=genre_names), file=f)
        print("\n", file=f)

        print("=== Accuracy Score ===", file=f)
        print(f'{accuracy_score(y_test, test_pred_class):.4f}', file=f)

        print("=== ROC-AUC (macro OvR) ===", file=f)
        print(f'{roc_auc_score(y_test, test_pred_proba, multi_class="ovr", average="macro"):.4f}', file=f)

        print("=== Top Features ===", file=f)
        # fixed: only call once
        top_lstm_features = get_top_lstm_features(model, X_test, vectorizer, f)

    cm = confusion_matrix(y_test, test_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['romance', 'horror', 'comedy', 'action'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('LSTM Confusion Matrix')
    plt.savefig(cm_file)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (genre, frequencies) in enumerate(top_lstm_features.items()):
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
        wc.generate_from_frequencies(frequencies)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'Top Words for {genre}', fontsize=16)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(wordcloud_file, dpi=300)
    plt.close()

    plot_model(
        model,
        to_file=arch_file,
        show_shapes=True,
        show_layer_names=True
    )


def evaluate_lstm_training(hist_lstm, paths):
    '''plots the LSTM training details'''
    lstm_metrics_dir = paths['metrics'] + "/lstm"

    loss_file = lstm_metrics_dir + "/lstm_model_loss.png"
    accuracy_file = lstm_metrics_dir + "/lstm_accuracy_over_epochs.png"

    plt.figure(figsize=(8, 5))
    plt.plot(hist_lstm.history['accuracy'], marker='o', label='Training Accuracy')
    plt.plot(hist_lstm.history['val_accuracy'], marker='x', label='Validation Accuracy')
    plt.title('LSTM Model Accuracy Over Epochs', fontsize=14)
    plt.xlabel('Epoch Number', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(0, len(hist_lstm.history['accuracy']), 5))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.savefig(accuracy_file, dpi=300)

    plt.figure()
    plt.plot(hist_lstm.history['loss'])
    plt.plot(hist_lstm.history['val_loss'])
    plt.title('LSTM Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(loss_file, dpi=300)