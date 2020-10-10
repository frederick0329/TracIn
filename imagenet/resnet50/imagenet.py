"""Get tf.dataset for Imagenet."""

from typing import Any, Text, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
import preprocessing

BUFFER_SIZE = 10000
NUM_TRAIN = 1281167
NUM_VAL = 50000
NUM_CLASSES = 1000
IMAGE_SIZE = 224


def _preprocess_train(image: tf.Tensor,
                      label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Apply image preprocessing."""
  image = preprocessing.preprocess_for_train(
      image,
      image_size=IMAGE_SIZE)
  label = tf.cast(label, tf.int32)
  return image, label


def _preprocess_eval(image: tf.Tensor,
                     label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Apply image preprocessing."""
  image = preprocessing.preprocess_for_eval(
      image,
      image_size=IMAGE_SIZE)
  label = tf.cast(label, tf.int32)
  return image, label


def _pipeline_ds(dataset: tf.data.Dataset,
                 batch_size: int,
                 is_training=False):
  """Preprocess and batch dataset."""
  if is_training:
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.repeat()
    dataset = dataset.map(_preprocess_train,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    dataset = dataset.map(_preprocess_eval,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size,
                          drop_remainder=is_training)

  # Prefetch overlaps in-feed with training
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def _make_get_dataset_fn(split: Text, batch_size: int, is_training: bool):
  """Make get_dataset_fn."""

  def get_dataset(
      input_context: tf.distribute.InputContext = None) -> tf.data.Dataset:
    builder = tfds.builder('imagenet2012')
    builder.download_and_prepare()

    read_config = tfds.ReadConfig(
        interleave_cycle_length=10,
        interleave_block_length=1,
        input_context=input_context)

    decoders = {'image': tfds.decode.SkipDecoding()}

    ds = builder.as_dataset(
        split=split,
        as_supervised=True,
        shuffle_files=is_training,
        decoders=decoders,
        read_config=read_config)
    ds = _pipeline_ds(ds, batch_size, is_training)
    return ds
  return get_dataset


def get_datasets(train_batch_size: int,
                 val_batch_size: int,
                 strategy: tf.distribute.Strategy) -> Tuple[Any, Any]:
  """Create and return train and validation dataset builders."""
  ds_train = strategy.experimental_distribute_datasets_from_function(
      _make_get_dataset_fn('train', train_batch_size, True))
  ds_val = strategy.experimental_distribute_datasets_from_function(
      _make_get_dataset_fn('validation', val_batch_size, False))

  return ds_train, ds_val
