import tensorflow as tf
import numpy as np
from kerastuner import RandomSearch

from .data_load import get_data, get_train_val_test_splits
from . import utils


def get_model(num_layers, num_dense_layers, type_layer, cell_size, learning_rate, dropout_rate):
    features_inp = tf.keras.layers.Input((utils.SEQ_EXTENTION, 768))
    x = features_inp
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    for i in range(num_layers):
        return_sequence = i < num_layers - 1
        if type_layer == "LSTM":
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cell_size, return_sequences=return_sequence))(x)
        else:
            x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(cell_size, return_sequences=return_sequence))(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    for i in range(num_dense_layers):
        x = tf.keras.layers.Dense(cell_size, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(features_inp, x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    model.summary()
    return model


def get_hp_model(hp):
    num_layers = hp.Int("num_layers", min_value=1, max_value=3)
    num_dense_layers = hp.Int("num_dense_layers", min_value=0, max_value=3)
    type_layer = hp.Choice("type_layer", values=["GRU", "LSTM"])
    cell_size = hp.Int("cell_size", min_value=16, max_value=1024, sampling="log")
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
    dropout_rate = hp.Choice("dropout_rate", values=[0.0, 0.001, 0.01, 0.1, 0.25, 0.5])

    return get_model(num_layers, num_dense_layers, type_layer, cell_size, learning_rate, dropout_rate)


def search_model():
    data = get_data()
    data_train, data_val_model, data_val_interpretation, data_test = get_train_val_test_splits(data)

    train_features = np.load("train_features.npy")
    valid_features = np.load("valid_features.npy")
    train_y = data_train["output"].values
    valid_y = data_val_model["output"].values
    tuner = RandomSearch(
        get_hp_model,
        objective='val_accuracy',
        max_trials=20,
        executions_per_trial=1,
        directory='test',
        project_name='test')

    tuner.search(train_features, y=train_y, batch_size=32, epochs=300,
                 validation_data=(valid_features, valid_y), verbose=2,
                 class_weight=dict(enumerate(utils.get_class_weights(train_y))),
                 # callbacks=[EarlyStopping(patience=30)]
                 )
    tuner.results_summary()