import typing
import warnings
import tensorflow as tf
import typing_extensions as tx

import layers, model_utils

def build_TFMaT(
    sequence_length: int,
    motif_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    classes: int,
    dropout=0.1,
    activation="linear",
    include_top=True,
    representation_size=None,
    motif_length_max=35
):

    x = tf.keras.layers.Input(shape=(sequence_length,4))
    motif_embedding = tf.keras.layers.Conv1D(
        filters=motif_size,
        kernel_size=motif_length_max,
        strides=1,
        padding="same",
        name="motif_embedding",
    )
    motif_embedding.trainable = False
    y=motif_embedding(x)
    y=tf.keras.layers.Dense(
        units=hidden_size,
        name="motif_to_hidden_embedding"
    )(y)
    y = layers.ClassToken(name="class_token")(y)
    y = layers.AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(num_layers):
        y, _ = layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
        )(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0],
                               name="ExtractToken")(y)

    y = tf.keras.layers.Dense(
        representation_size,
        name="pre_logits",
        activation="tanh"
    )(y)
    if include_top:
        y = tf.keras.layers.Dense(
            classes,
            name="head",
            activation=activation)(y)
    return tf.keras.models.Model(inputs=x, outputs=y, name=name)
