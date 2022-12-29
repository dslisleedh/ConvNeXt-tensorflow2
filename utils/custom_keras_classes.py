import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import numpy as np

import gin

from functools import partial


# Don't use this with EarlyStopping.
@gin.configurable
class SwapAverageWeights(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        optimizer_name = self.model.optimizer._name
        assert optimizer_name in ['MovingAverage', 'SWA'], \
            'Optimizer must be a MovingAverage or SWA optimizer, got {}'.format(optimizer_name)

    def on_test_begin(self, logs=None):
        self.shallow_weights = self.model.get_weights()
        self.model.optimizer.assign_average_vars(self.model.trainable_variables)

    def on_test_end(self, logs=None):
        self.model.set_weights(self.shallow_weights)

    def on_train_end(self, logs=None):
        self.model.optimizer.assign_average_vars(self.model.trainable_variables)


@gin.configurable
class SwapAverageWeightsEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, *args, **kwargs):
        super(SwapAverageWeightsEarlyStopping, self).__init__(*args, **kwargs)

    def on_train_begin(self, logs=None):
        optimizer_name = self.model.optimizer._name
        assert optimizer_name in ['MovingAverage', 'SWA'], \
            'Optimizer must be a MovingAverage or SWA optimizer, got {}'.format(optimizer_name)
        super(SwapAverageWeightsEarlyStopping, self).on_train_begin(logs)

    def on_test_begin(self, logs=None):
        self.shallow_weights = self.model.get_weights()
        self.model.optimizer.assign_average_vars(self.model.trainable_variables)

    def on_epoch_end(self, epoch, logs=None):
        super(SwapAverageWeightsEarlyStopping, self).on_epoch_end(epoch, logs)
        if not self.model.stop_training:
            self.model.set_weights(self.shallow_weights)


def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold=0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    learning_rate = 0.5 * target_lr * (
                1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)

    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


@gin.configurable
class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, hold=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_train_begin(self, logs=None):
        self.global_step = 0
        self.target_lr = K.get_value(self.model.optimizer.lr)

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr)

