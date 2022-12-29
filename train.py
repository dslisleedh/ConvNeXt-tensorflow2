import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from src.model import ConvNeXt

import gin
from omegaconf import OmegaConf
import hydra
from hydra.utils import get_original_cwd

import tensorflow_datasets as tfds

from typing import Tuple, List, Union, Optional, Callable, Sequence
from functools import partial

from utils.configure import external_configurable
from utils.custom_keras_classes import *
from utils.preprocessing import preprocessing
from utils.system import Runner


@gin.configurable
def train(
    model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss, metrics: List[tf.keras.metrics.Metric],
    epochs: int, warmup_epochs: int, batch_size: int, patience: int,
    use_ema: bool, average_decay: float, image_size: Sequence[int]
):
    with open('./config.gin', 'w') as f:
        f.write(gin.operative_config_str())

    with tf.device('/cpu:0'):
        print('\nLoading dataset...')
        train_ds, valid_ds, test_ds = tfds.load(
            'plant_village', as_supervised=True, split=['train[:60%]', 'train[60%:80%]', 'train[80%:]']
        )
        train_preprocessing = partial(preprocessing, augment=True, size=image_size)
        inference_preprocessing = partial(preprocessing, augment=False, size=image_size)
        train_ds = train_ds.shuffle(10000).map(train_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        valid_ds = valid_ds.map(inference_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.map(inference_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)\
            .batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    print('Start training...')
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs', update_freq='batch'
        ),
        WarmupCosineDecay(
            total_steps=epochs * train_ds.cardinality().numpy(),
            warmup_steps=warmup_epochs * train_ds.cardinality().numpy()
        ),
    ]
    if patience > 0:
        es_kwargs = {
            'monitor': 'val_loss', 'mode': 'min', 'patience': patience, 'restore_best_weights': True
        }
        if use_ema:
            callbacks.append(SwapAverageWeightsEarlyStopping(**es_kwargs))
        else:
            callbacks.append(tf.keras.callbacks.EarlyStopping(**es_kwargs))
    else:
        if use_ema:
            callbacks.append(SwapAverageWeights())

    if use_ema:
        optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=average_decay)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    model.fit(
        train_ds, epochs=epochs, validation_data=valid_ds, callbacks=callbacks
    )

    print('\nStart evaluating...')
    result = model.evaluate(test_ds)

    print('\nTrain Result:')
    with open('./result.txt', 'w') as f:
        for metric, value in zip(model.metrics_names, result):
            print(f'{metric}: {value}')
            f.write(f'{metric}: {value} \n')

    print('\nSaving model...')
    model.save_weights('./model_weights')


@hydra.main(config_path='./conf', config_name='config', version_base=None)
def main(main_config):
    # To prevent gin from load the config multiple times when use --multirun which will raise an error
    @Runner
    def _main():
        external_configurable()
        config_files = [
            get_original_cwd() + '/conf/model_config/' + main_config.model + '.gin',
            get_original_cwd() + '/conf/hyper_params.gin',
        ]
        gin.parse_config_files_and_bindings(config_files, None)
        train()

    _main()


if __name__ == '__main__':
    main()
