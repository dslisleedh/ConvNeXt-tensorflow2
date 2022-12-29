import tensorflow as tf
import tensorflow_addons as tfa

from typing import Tuple, List, Union, Optional, Callable, Sequence


def global_init(shape, dtype=None):
    return tf.random.truncated_normal(shape, stddev=0.02, dtype=dtype)


class PatchifySTEM(tf.keras.layers.Layer):
    def __init__(self, n_filters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_filters = n_filters
        self.conv = tf.keras.layers.Conv2D(
            filters=n_filters, kernel_size=(4, 4), strides=(4, 4), padding='VALID',
            kernel_initializer=global_init
        )
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        x = self.ln(x)
        return x


class DownSample(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv = tf.keras.layers.Conv2D(
            filters=n_filters, kernel_size=(2, 2), strides=(2, 2), padding='VALID',
            kernel_initializer=global_init
        )

    def call(self, inputs, *args, **kwargs):
        x = self.ln(inputs)
        x = self.conv(x)
        return x


class ConvNeXtBlock(tf.keras.layers.Layer):
    def __init__(
            self, dwc_kernel: Sequence[int] = (7, 7), expansion_rate: int = 4, act: Callable = tf.nn.gelu,
            norm: Callable = tf.keras.layers.LayerNormalization, scale_init: float = 1e-6, drop_rate: float = 0.0,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dwc_kernel = dwc_kernel
        self.expansion_rate = expansion_rate
        self.act = act
        self.norm = norm(epsilon=1e-6)
        self.scale_init = scale_init
        self.drop_rate = float(drop_rate)

    def build(self, input_shape):
        self.dwc = tf.keras.layers.DepthwiseConv2D(
            kernel_size=self.dwc_kernel, strides=(1, 1), padding='SAME', kernel_initializer=global_init
        )
        self.up_conv = tf.keras.layers.Dense(
            units=input_shape[-1] * self.expansion_rate, kernel_initializer=global_init
        )
        self.down_conv = tf.keras.layers.Dense(
            units=input_shape[-1], kernel_initializer=global_init
        )
        self.gamma = self.add_weight(
            name='gamma', shape=(1, 1, 1, input_shape[-1]), initializer=tf.keras.initializers.Constant(self.scale_init),
            trainable=True
        )
        self.stochastic_depth = tfa.layers.StochasticDepth(self.drop_rate)

    def call(self, inputs, *args, **kwargs):
        x = self.dwc(inputs)
        x = self.norm(x)
        x = self.up_conv(x)
        x = self.act(x)
        x = self.down_conv(x)
        return self.stochastic_depth([inputs, x * self.gamma])


# LATER: This raises error when jit-compiling. Fix this later.
#        For now, use tfa.layers.StochasticDepth instead.
# class DropPath(tf.keras.layers.Layer):
#     def __init__(
#          self, forward: Callable, rate: float
#     ):
#         super(DropPath, self).__init__()
#         self.forward = forward
#         self.rate = rate
#
#     def call(self, inputs, training: bool = False, *args, **kwargs):
#         if not training or (self.rate == 0.):
#             return self.forward(inputs)
#
#         keep_prob = 1. - self.rate
#         if tf.random.uniform((), minval=0., maxval=1.) < keep_prob:
#             inputs = inputs + ((self.forward(inputs) - inputs) / keep_prob)
#         return inputs


class ClassificationHead(tf.keras.layers.Layer):
    def __init__(
            self, n_classes: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes

    def build(self, input_shape):
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.classifier = tf.keras.layers.Dense(
            units=self.n_classes, kernel_initializer=tf.keras.initializers.zeros,
            activation=tf.nn.softmax
        )

    def call(self, inputs, *args, **kwargs):
        x = tf.reduce_mean(inputs, axis=(1, 2))
        x = self.ln(x)
        return self.classifier(x)
