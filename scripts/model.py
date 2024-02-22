import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    ## Build the model

    Our model processes a tensor of shape `(batch size, sequence length, features)`,
    where `sequence length` is the number of time steps and `features` is each input
    timeseries.

    We include residual connections, layer normalization, and dropout.
    The resulting layer can be stacked multiple times.

    The projection layers are implemented through `keras.layers.Conv1D`.

    """
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    ):
    """
    The main part of our model is now complete. We can stack multiple of those
    `transformer_encoder` blocks and we can also proceed to add the final
    Multi-Layer Perceptron classification head. Apart from a stack of `Dense`
    layers, we need to reduce the output tensor of the `TransformerEncoder` part of
    our model down to a vector of features for each data point in the current
    batch. A common way to achieve this is to use a pooling layer. For
    this example, a `GlobalAveragePooling1D` layer is sufficient.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)
