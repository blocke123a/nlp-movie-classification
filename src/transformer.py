'''
transformer.py

Train and save a transformer model on the given dataframe
'''
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import (MultiHeadAttention, LayerNormalization,
                                     GlobalAveragePooling1D, Add)
from .utils import make_save_emb_layer, get_embeddings, get_transformer_top_features
from wordcloud import WordCloud


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


def make_save_emb_layer(word_index, embeddings_index, layer_file_name):
    '''make and save the embedding layer'''
    embedding_matrix, unknown = get_embedding_matrix(word_index, embeddings_index)
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)
    with open(layer_file_name, 'wb') as f:
        pickle.dump(embedding_layer, f, -1)
    return unknown


def transformer_encoder_block(x, num_heads, ff_dim, dropout_rate=0.1):
    # Multi-Head Self-Attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1] // num_heads)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))  # residual + norm

    # Feed-Forward Network
    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(x.shape[-1])(ffn)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn])) # residual + norm
    return x


def get_transformer_model(params, num_classes=4, num_heads=4, ff_dim=128, num_transformer_blocks=2, dropout_rate=0.1, embeddings_dim=300):
    input_layer = Input(shape=(params["MAX_SEQUENCE_LENGTH"],))

    x = embedding_layer(input_layer) # (batch, seq_len, 300)
    x = PositionalEncoding(max_len=params["MAX_SEQUENCE_LENGTH"], d_model=embeddings_dim)(x)
    x = Dropout(dropout_rate)(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, num_heads=num_heads, ff_dim=ff_dim,
                                       dropout_rate=dropout_rate)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

def train_transformer(df: pd.DataFrame, cfg: dict):
    '''train and save a transformer model'''

    params = cfg.get("model_parameters", {})
    paths = cfg.get("paths", {})
    # get model paths
    nb_model_dir = paths['models']
    nb_metrics_dir = paths['metrics'] + "/transformer"
    model_file = nb_model_dir + "/transformer_genre_model.keras"
    weights_file = nb_model_dir + "/best_transformer_model.weights.h5"
    tokenizer_file = nb_model_dir + "keras_tokenizer.pkl"
    layer_file = nb_model_dir + "embedding_layer.pkl"

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

        transformer_model = get_transformer_model(params)

        t_early_stopping = EarlyStopping(patience=12, restore_best_weights=True)
        t_model_checkpoint = ModelCheckpoint(weights_file,
                                            save_best_only=True, save_weights_only=True,
                                            monitor='val_accuracy', mode='max')

        hist_transformer = transformer_model.fit(
            X_train_lstm, y_train,
            validation_data=(X_test_lstm, y_test),
            epochs=100, batch_size=params['BATCH_SIZE'], shuffle=True, verbose=2,
            callbacks=[t_model_checkpoint, t_early_stopping]
        )
        transformer_model.load_weights(weights_file)
        transformer_model.save(model_file)

        evaluate_transformer_training(hist_transformer, paths)

    else:
        # load existing model
        model = model = load_model(model_file, custom_objects={'PositionalEncoding': PositionalEncoding})

    # evaluate the model
    evaluate_transformer(model, X_test_lstm, y_test, paths, params)



def evaluate_transformer(model, vectorizer, X_test, y_test, paths, params):
    '''evaluates the naive bayes model'''
    trans_metrics_dir = paths['metrics'] + "/transformer"

    test_pred_proba = model.predict(X_test, batch_size=params['BATCH_SIZE'], verbose=0)
    test_pred_class = np.argmax(test_pred_proba, axis=1)

    genre_names = ['romance', 'horror', 'comedy', 'action']

    evaluation_file = trans_metrics_dir + "/transformer_evaulation.txt"
    cm_file = trans_metrics_dir + "/transformer_cm.jpg"
    arch_file = trans_metrics_dir + "transformer_model_architecture.png"
    wordcloud_file = trans_metrics_dir + "/transformer_wordcloud.jpg"

    with open(evaluation_file, "w", encoding='utf-8') as f:
        print("====== Model Evaluation: Transformer ======\n", file=f)

        #classification report
        print("=== Classification Report ===", file=f)
        print(classification_report(y_test, test_pred_class, target_names=genre_names), file=f)
        print("\n", file=f)

        print("=== Accuracy Score ===", file=f)
        print(f'{accuracy_score(y_test, test_pred_class):.4f}', file=f) #get accuracy

        print("=== ROC-AUC (macro OvR) ===", file=f)
        print(f'{roc_auc_score(y_test, test_pred_proba, multi_class="ovr", average="macro"):.4f}', file=f)

        print("\n=== Top Features for the Transformer model ===\n", file=f)
        #get top features
        genre_word_frequencies = get_transformer_top_features(model, vectorizer, n=40)
        
        for genre, frequencies in genre_word_frequencies.items():
            print(f"Top Driving Words for {genre}:", file=f)
            #get top 10 for summary
            top_10 = list(frequencies.items())[:10]
            for word, score in top_10:
                print(f"  - {word}: {score:.6f}", file=f)
            print("-" * 30, file=f)

    cm = confusion_matrix(y_test, test_pred_class)

    disp = (ConfusionMatrixDisplay(confusion_matrix=cm, 
                    display_labels=['romance', 'horror','comedy','action']))
    disp.plot(cmap=plt.cm.Purples)
    plt.title('Transformer Confusion Matrix')
    plt.savefig(cm_file)

    # plot & save model architecture
    plot_model(
        model,
        to_file=arch_file,
        show_shapes=True,
        show_layer_names=True
    )

    #wordcloud
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (genre, frequencies) in enumerate(genre_word_frequencies.items()):
        wc = WordCloud(background_color='white', width=800, height=400, max_words=40)
        wc.generate_from_frequencies(frequencies)
        
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'Transformer: Top Words for {genre}', fontsize=16)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(wordcloud_file, dpi=300)
    plt.close()


def evaluate_transformer_training(hist_transformer, paths):
    '''plots the lstm training details'''
    trans_metrics_dir = paths['metrics'] + "/transformer"

    loss_file = trans_metrics_dir + "/transformer_model_loss.png"
    accuracy_file = trans_metrics_dir + "/transformer_accuracy_over_epochs.png"
    # plot training and validation accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(hist_transformer.history['accuracy'], marker='o', label='Training Accuracy')
    plt.plot(hist_transformer.history['val_accuracy'], marker='x', label='Validation Accuracy')

    # format plot
    plt.title('Transformer Model Accuracy Over Epochs', fontsize=14)
    plt.xlabel('Epoch Number', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(0, len(hist_transformer.history['accuracy']), 5))  # ensures integer epoch numbers on x-axis
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    # save and display plot
    plt.savefig(accuracy_file, dpi=300)

    plt.plot(hist_transformer.history['loss'])
    plt.plot(hist_transformer.history['val_loss'])
    plt.title('Transformer Loss over Epochs')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.savefig(loss_file, dpi=300)