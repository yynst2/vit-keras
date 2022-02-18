import tensorflow as tf
# from vit_keras import vit, utils
from test.notebook_test_run.TFMaT import *
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical

model = build_TFMaT(
    name='TFMat_test',
    sequence_length=1000,
    motif_size=6000,
    num_layers=12,
    num_heads=12,
    hidden_size=768,
    mlp_dim=3072,
    classes=1000,
    representation_size=768,
    motif_length_max=35,
    include_top=True,
    activation='sigmoid'
)

#
optimizer = tf.keras.optimizers.Adam()
# need compile
model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
    )


def input_fn():
    batch_size = 64
    dna_seq = np.random.random(size=(128, 1, 1000, 4))
    dna_seq_exp = np.expand_dims(dna_seq, axis=-1)
    labels = np.random.randint(low=0, high=1000, size=(128, 1))
    labels_cat = to_categorical(labels, num_classes=1000)
    labels_cat = np.expand_dims(labels_cat, 1)

    dataset = Dataset.from_tensor_slices((dna_seq, labels_cat))
    dataset.shuffle(128, reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    return dataset

#
params={}
params['batch_size']=64
est = tf.keras.estimator.model_to_estimator(keras_model=model)

est.train(
input_fn=input_fn,
max_steps=10,
)