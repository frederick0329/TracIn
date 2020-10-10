# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs an Image Classification model."""

import os
from typing import Any, Text, List, Mapping

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

import distribution_utils
import imagenet
import learning_rate
import resnet


flags.DEFINE_string(
    'model_dir', None,
    'Directory to save checkpoints and logs.')
flags.mark_flag_as_required('model_dir')

flags.DEFINE_string(
    'distribution_strategy', None,
    'Distribution strategy. [one_device, tpu]')
flags.mark_flag_as_required('distribution_strategy')

flags.DEFINE_string(
    'tpu', None,
    'TPU address.')

flags.DEFINE_integer(
    'train_batch_size', 128,
    'Batch size for training (per replica).')

flags.DEFINE_integer(
    'val_batch_size', 128,
    'Batch size for eval. (per replica)')

flags.DEFINE_integer(
    'train_epochs', 90,
    'Number of epochs to train for.')

flags.DEFINE_float(
    'warmup_epochs', 5,
    'Number of epochs of warmup.')

flags.DEFINE_list(
    'boundaries', [30, 60, 80],
    'Boundaries for learning rate decay.')

flags.DEFINE_list(
    'multipliers', [1.0, 0.1, 0.01, 0.001],
    'Multipliers for learning rate decay.')


def _get_metrics() -> List[Any]:
  """Get a list of metrics to track."""
  return [
      tf.keras.metrics.SparseCategoricalAccuracy(name='top_1_accuracy'),
      tf.keras.metrics.SparseTopKCategoricalAccuracy(
          k=5, name='top_5_accuracy'),
  ]


def _get_optimizer(batch_size: int, epoch_size: int, warmup_epochs: int,
                   boundaries: List[int], multipliers: List[float],
                   momentum: float, nesterov: bool = False
                   ) -> tf.keras.optimizers.Optimizer:
  """Get optimizer."""
  lr = learning_rate.PiecewiseConstantDecayWithWarmup(
      batch_size=batch_size,
      epoch_size=epoch_size,
      warmup_epochs=warmup_epochs,
      boundaries=boundaries,
      multipliers=multipliers)
  optimizer = tf.keras.optimizers.SGD(learning_rate=lr,
                                      momentum=momentum,
                                      nesterov=nesterov)
  return optimizer


def _get_callbacks(model_dir: Text):
  """Get callbacks for model.fit."""
  tf.io.gfile.makedirs(model_dir)
  ckpt_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
  callbacks = [
      tf.keras.callbacks.TensorBoard(log_dir=model_dir),
      tf.keras.callbacks.ModelCheckpoint(
          filepath=ckpt_path,
          save_weights_only=True)
  ]
  return callbacks


def initialize():
  """Initializes backend related initializations."""
  if tf.config.list_physical_devices('GPU'):
    data_format = 'channels_first'
  else:
    data_format = 'channels_last'
  tf.keras.backend.set_image_data_format(data_format)


def train_and_eval(flags_obj: flags.FlagValues) -> Mapping[str, Any]:
  """Runs the train and eval path using compile/fit."""
  logging.info('Running train and eval.')

  # Note: for TPUs, strategy and scope should be created before the dataset
  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=0,
      tpu_address=flags_obj.tpu)

  num_replicas = strategy.num_replicas_in_sync
  global_train_batch_size = flags_obj.train_batch_size * num_replicas
  global_val_batch_size = flags_obj.val_batch_size * num_replicas

  logging.info('Detected %d devices.',
               strategy.num_replicas_in_sync if strategy else 1)

  ds_train, ds_val = imagenet.get_datasets(
      flags_obj.train_batch_size,
      flags_obj.val_batch_size,
      strategy)

  train_epochs = flags_obj.train_epochs
  train_steps = imagenet.NUM_TRAIN // global_train_batch_size
  validation_steps = imagenet.NUM_VAL // global_val_batch_size

  initialize()

  with strategy.scope():
    model = resnet.resnet50(imagenet.NUM_CLASSES)
    optimizer = _get_optimizer(global_train_batch_size,
                               train_steps * global_train_batch_size,
                               flags_obj.warmup_epochs,
                               flags_obj.boundaries,
                               flags_obj.multipliers,
                               momentum=0.9,
                               nesterov=False)
    metrics = _get_metrics()
    callbacks = _get_callbacks(flags_obj.model_dir)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  experimental_steps_per_execution=train_steps)
    model.fit(
        ds_train,
        epochs=train_epochs,
        steps_per_epoch=train_steps,
        verbose=2,
        validation_data=ds_val,
        validation_steps=validation_steps,
        callbacks=callbacks)


def main(_):
  train_and_eval(flags.FLAGS)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
