import tensorflow as tf
import tensorflow_addons as tfa

from typing import Tuple, List, Union, Optional, Callable, Sequence
import gin

from src.layers import *


@gin.configurable
class ConvNeXt(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config

        drop_rates = tf.linspace(0.0, config['drop_rate'], sum(config['n_blocks']))
        self.forward = tf.keras.Sequential()
        for i, (n_filters, n_blocks) in enumerate(zip(config['n_filters'], config['n_blocks'])):
            if i == 0:
                self.forward.add(PatchifySTEM(n_filters))
            else:
                self.forward.add(DownSample(n_filters))
            for n in range(n_blocks):
                self.forward.add(
                    ConvNeXtBlock(**config['block'], drop_rate=drop_rates[sum(config['n_blocks'][:i]) + n])
                )

        self.forward.add(ClassificationHead(config['n_classes']))

    @tf.function(jit_compile=True)
    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)
