""" A wrapper for the Caltech-UCSD Birds-200 2011 dataset
"""


import os
import numpy as np
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'birds_%s_*.tfrecord'
SPLITS_TO_SIZES = {'train': 5994, 'test': 5794}

_NUM_CLASSES = 200

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 1 and 200',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('Split name %s was not recognised.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  if reader is None:
    reader = tf.TFRecordReader  # should not call this constructor

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
      'image/format': tf.FixedLenFeature([], tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = dict(np.genfromtxt(os.path.join(dataset_dir, 'classes.txt'),
                                       dtype=None, encoding=None))

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names
  )
