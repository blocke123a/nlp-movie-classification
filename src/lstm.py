'''
naive_bayes.py

Train and pickle a naive bayes model on the given dataframe
'''
import os
import pickle
import zipfile
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Input, Embedding, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from .utils import make_save_emb_layer, get_embeddings


def get_model(params):
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
    # 4 output neurons with softmax for multiclass
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
    # get model paths
    lstm_model_dir = paths['models']
    lstm_metrics_dir = paths['metrics'] + "/lstm"
    model_file = lstm_model_dir + "/lstm_genre_model.keras"
    weights_file = lstm_model_dir + "/best_lstm_model.weights.h5"
    tokenizer_file = lstm_model_dir + "keras_tokenizer.pkl"
    layer_file = lstm_model_dir + "embedding_layer.pkl"

    # make directories
    os.makedirs(paths['models'], exist_ok=True)
    os.makedirs(paths['metrics'], exist_ok=True)
    os.makedirs(lstm_model_dir, exist_ok=True)
    os.makedirs(lstm_metrics_dir, exist_ok=True)


    X = df['Final_Summary'] # create X
    y = df['genre_map'] # create y

    # make train/test split
    X_train, X_test, y_train, y_test = (train_test_split(X, y, 
                                        test_size=params['test_split'],
                                        random_state=params['random_state']))
    
    # train model if it doesn't exist already
    tokenizer = Tokenizer(filters="")
    tokenizer.fit_on_texts(X_train)

    with open(tokenizer_file, 'wb') as f:
        pickle.dump(tokenizer, f)

    word_index = tokenizer.word_index
    
    MAX_SEQUENCE_LENGTH = params['MAX_SEQUENCE_LENGTH']

    # turn text into sequence of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # pad sequences to the same length
    X_train_lstm = pad_sequences(X_train_seq,maxlen=MAX_SEQUENCE_LENGTH,padding='post',
                            truncating='post')
    X_test_lstm = pad_sequences(X_test_seq,maxlen=MAX_SEQUENCE_LENGTH,padding='post',
                            truncating='post')
        
        # adjust class labels to start at 0 to match index instead of 1
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
        lstm_model = get_model(params)

        hist_lstm = lstm_model.fit(
            X_train_lstm, y_train,
            validation_data=(X_test_lstm, y_test),
            epochs=50, batch_size=params['BATCH_SIZE'], shuffle=True, verbose=2,
            callbacks=[model_checkpoint, early_stopping]
        )
        lstm_model.load_weights(weights_file)

        lstm_model.save(model_file)
        evaluate_lstm_training(hist_lstm, paths)

    else:
        # load existing model
        model = load_model(model_file)

    # evaluate the model
    evaluate_lstm(model, X_test_lstm, y_test, paths, params)



def evaluate_lstm(model, X_test, y_test, paths, params):
    '''evaluates the logistic regression model'''
    lstm_metrics_dir = paths['metrics'] + "/lstm"

    test_pred_proba = model.predict(X_test, batch_size=params['BATCH_SIZE'], verbose=0)
    test_pred_class = np.argmax(test_pred_proba, axis=1)

    genre_names = ['romance', 'horror', 'comedy', 'action']

    evaluation_file = lstm_metrics_dir + "/lstm_evaulation.txt"
    cm_file = lstm_metrics_dir + "/lstm_cm.jpg"
    arch_file = lstm_metrics_dir + "lstm_model_architecture.png"

    with open(evaluation_file, "w", encoding='utf-8') as f:
        print("====== Model Evaluation: LSTM ======\n", file=f)

        #classification report
        print("=== Classification Report ===", file=f)
        print(classification_report(y_test, test_pred_class, target_names=genre_names), file=f)
        print("\n", file=f)

        print("=== Accuracy Score ===", file=f)
        print(f'{accuracy_score(y_test, test_pred_class):.4f}', file=f) #get accuracy

        print("=== ROC-AUC (macro OvR) ===", file=f)
        print(f'{roc_auc_score(y_test, test_pred_proba, multi_class="ovr", average="macro"):.4f}', file=f)

    cm = confusion_matrix(y_test, test_pred_class)

    disp = (ConfusionMatrixDisplay(confusion_matrix=cm, 
                    display_labels=['romance', 'horror','comedy','action']))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('LSTM Confusion Matrix')
    plt.savefig(cm_file)

    # plot & save model architecture
    plot_model(
        model,
        to_file=arch_file,
        show_shapes=True,
        show_layer_names=True
    )


def evaluate_lstm_training(hist_lstm, paths):
    '''plots the lstm training details'''
    lstm_metrics_dir = paths['metrics'] + "/lstm"

    loss_file = lstm_metrics_dir + "/lstm_model_loss.png"
    accuracy_file = lstm_metrics_dir + "/lstm_accuracy_over_epochs.png"
    # plot training and validation accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(hist_lstm.history['accuracy'], marker='o', label='Training Accuracy')
    plt.plot(hist_lstm.history['val_accuracy'], marker='x', label='Validation Accuracy')

    # format plot
    plt.title('LSTM Model Accuracy Over Epochs', fontsize=14)
    plt.xlabel('Epoch Number', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(0, len(hist_lstm.history['accuracy']), 5))  # ensures integer epoch numbers on x-axis
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    # save and display plot
    plt.savefig(accuracy_file, dpi=300)

    plt.plot(hist_lstm.history['loss'])
    plt.plot(hist_lstm.history['val_loss'])
    plt.title('model loss by AUC score')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.savefig(loss_file, dpi=300)