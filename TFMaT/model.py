import sys,os
import argparse

sys.path.append('/ocean/projects/bio220011p/xcheni')
from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec
import tensorflow as tf
import layers
# import model_utils # ImportError: libGL.so.1 # todo


# model_fn
def model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=None):
    # x = tf.keras.layers.Input(shape=(params['sequence_length'],4))
    model_params=params['model']
    motif_embedding = tf.keras.layers.Conv1D(
        filters=model_params['motif_size'],
        kernel_size=model_params['motif_length_max'],
        strides=1,
        padding="same",
        name="motif_embedding",
    )
    motif_embedding.trainable = model_params['motif_embedding_trainable']
    y = motif_embedding(features)
    y = tf.keras.layers.Dense(
        units=model_params['hidden_size'],
        name="motif_to_hidden_embedding"
    )(y)
    y = layers.ClassToken(name="class_token")(y)
    y = layers.AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(model_params['num_layers']):
        y, _ = layers.TransformerBlock(
            num_heads=model_params['num_heads'],
            mlp_dim=model_params['mlp_dim'],
            dropout=model_params['dropout'],
            name=f"Transformer/encoderblock_{n}",
        )(y,True)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0],
                               name="ExtractToken")(y)

    y = tf.keras.layers.Dense(
        model_params['representation_size'],
        name="pre_logits",
        activation="tanh"
    )(y)

    logits = tf.keras.layers.Dense(
        model_params['classes'],
        name="head",
        activation=model_params['activation'])(y)

    learning_rate = tf.constant(model_params["lr"])
    loss_op = None
    train_op = None

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss_op = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        )
    train_op = tf.compat.v1.train.AdamOptimizer(
        learning_rate=learning_rate
    ).minimize(loss_op, global_step=tf.compat.v1.train.get_global_step())
    spec = CSEstimatorSpec(mode=mode, loss=loss_op, train_op=train_op)


    return spec