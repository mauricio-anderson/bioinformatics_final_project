""" """
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Embedding, Bidirectional,
    Dropout, Activation, BatchNormalization
)
from keras.regularizers import l2

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def create_model(vocab_size: int, max_sequence_len: int) -> Sequential:
    """ """

    # model based on:  Building and Training the Char-RNN Model (p581) /
    #   Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

    # GRU activation -> https://stackoverflow.com/questions/68844792/lstm-will-not-use-cudnn-kernels-since-it-doesnt-meet-the-criteria-it-will-use

    # CONFIG

    embedding_output_dim = 64

    gru_units = 128
    gru_dropout = 0.3
    # gru_recurrent_dropout = 0.3

    dropout_rate = 0.2
    add_bn = True
    hidden_layer_config = [
        # {"units": 100, "activation": "relu", "dropout_rate": dropout_rate},
        {"units": 50, "activation": "relu", "dropout_rate": dropout_rate},
        ]

    output_layer_activation = None  # "linear"

    # MODEL
    model = Sequential()

    # Embedding layer
    model.add(
       Embedding(
            input_dim = vocab_size + 1,
            output_dim = embedding_output_dim,
            input_length = max_sequence_len
        )
    )

    # Recurrent layer
    model.add(
        Bidirectional(
            GRU(
                gru_units,
                return_sequences=True,
                activation="tanh",
                dropout=gru_dropout,
                # recurrent_dropout=gru_recurrent_dropout,
                )
        )
    )

    model.add(
        Bidirectional(
            GRU(
                gru_units,
                return_sequences=False,
                activation="tanh",
                dropout=gru_dropout,
                # recurrent_dropout=gru_recurrent_dropout,
                )
        )
    )

    # Top fully-connected layer
    for layer in hidden_layer_config:
        dropout_rate = layer.get("dropout_rate", 0)
        if dropout_rate:
            model.add(Dropout(dropout_rate))
        model.add(Dense(layer["units"]))
        if layer.get("add_bn", add_bn):
            model.add(BatchNormalization())  # util para redes profundas
        model.add(Activation(layer["activation"]))

    # Output layer
    model.add(Dense(1, activation=output_layer_activation))

    return model
