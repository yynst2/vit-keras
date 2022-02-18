
"""
todo: this script needs to be adapted to TFMaT so we can actual train the model
"""
import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical


def _parse_batch_samples(samples):
    images = tf.image.convert_image_dtype(samples["image"], tf.float16,)
    batch_size = images.shape[0]
    images = tf.reshape(images, [batch_size, -1])
    labels = tf.cast(samples["label"], tf.int32)
    return images, labels


def input_fn(params, mode=tf.estimator.ModeKeys.TRAIN): # todo: adapted to TFMaT validate only mode

    batch_size = 128
    dna_seq = np.random.random(size=(12800, 1, 1000, 4))
    dna_seq_exp = np.expand_dims(dna_seq, axis=-1)
    labels = np.random.randint(low=0, high=1000, size=(12800, 1))
    labels_cat = to_categorical(labels, num_classes=1000)
    labels_cat = np.expand_dims(labels_cat, 1)

    dataset = Dataset.from_tensor_slices((dna_seq, labels_cat))
    dataset.shuffle(128).repeat().batch(batch_size, drop_remainder=True)
    dataset=dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def train_input_fn(params):
    return input_fn(params, mode=tf.estimator.ModeKeys.TRAIN)


def eval_input_fn(params):
    return input_fn(params, mode=tf.estimator.ModeKeys.EVAL)
