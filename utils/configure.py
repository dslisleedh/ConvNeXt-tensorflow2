import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np

from typing import Tuple, List, Union, Optional, Callable, Sequence
from functools import partial

import gin


def external_configurable():
    import gin.tf.external_configurables

    # Losses
    def _register_losses(module):
        gin.config.external_configurable(module, module='tf.keras.losses')
    _register_losses(tf.keras.losses.MeanSquaredError)
    _register_losses(tf.keras.losses.MeanAbsoluteError)
    _register_losses(tf.keras.losses.BinaryCrossentropy)
    _register_losses(tf.keras.losses.SparseCategoricalCrossentropy)
    _register_losses(tf.keras.losses.CategoricalCrossentropy)

    # Metrics
    def _register_metrics(module):
        gin.config.external_configurable(module, module='tf.keras.metrics')
    _register_metrics(tf.keras.metrics.BinaryAccuracy)
    _register_metrics(tf.keras.metrics.CategoricalAccuracy)
    _register_metrics(tf.keras.metrics.SparseCategoricalAccuracy)
    gin.config.external_configurable(tfa.metrics.F1Score, 'tfa.metrics.F1Score')

    # Activations
    gin.config.external_configurable(
        tf.nn.gelu, 'tf.nn.gelu'
    )
    gin.config.external_configurable(
        tf.nn.relu, 'tf.nn.relu'
    )

    # Layer
    gin.config.external_configurable(
        tf.keras.layers.BatchNormalization, 'tf.keras.layers.BatchNormalization'
    )
    gin.config.external_configurable(
        tf.keras.layers.LayerNormalization, 'tf.keras.layers.LayerNormalization'
    )

    # Optimizers
    gin.config.external_configurable(
        tf.keras.optimizers.Adam, 'tf.keras.optimizers.Adam'
    )
    gin.config.external_configurable(
        tfa.optimizers.AdamW, 'tfa.optimizers.AdamW'
    )
    gin.config.external_configurable(
        tfa.optimizers.MovingAverage, 'tfa.optimizers.MovingAverage'
    )