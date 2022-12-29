import tensorflow as tf

from typing import Sequence


def preprocessing(
        image, label, size: Sequence[int], augment: bool = False, n_classes: int = 38
):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, n_classes)

    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, size)
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    image = tf.squeeze(image, axis=0)
    return image, label
