# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts Oxford Flowers-102 data to TFRecords of TF-Example protos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import urllib
import scipy.io

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
_LABELS_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
_SPLIT_URL = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

CLASS_NAMES = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells',
    'sweet pea', 'english marigold', 'tiger lily', 'moon orchid',
    'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
    "colt's foot", 'king protea', 'spear thistle', 'yellow iris',
    'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
    'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary',
    'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
    'stemless gentian', 'artichoke', 'sweet william', 'carnation',
    'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
    'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip',
    'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia',
    'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
    'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
    'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
    'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone',
    'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum',
    'spring crocus', 'bearded iris', 'windflower', 'tree poppy', 'gazania',
    'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory',
    'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani',
    'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow',
    'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ',
    'bee balm', 'ball moss', 'foxglove', 'bougainvillea', 'camellia', 'mallow',
    'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper',
    'blackberry lily']


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(
        self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _download_file(url, file_path):
  """ Download a file from the given URL to a path """
  urllib.request.urlretrieve(url, file_path)


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of image filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  # directories = []
  # class_names = []
  # for filename in os.listdir(images_root):
  #   path = os.path.join(images_root, filename)
  #   if os.path.isdir(path):
  #     directories.append(path)
  #     class_names.append(filename)

  # photo_filenames = []
  # for directory in directories:
  #   for filename in os.listdir(directory):
  #     path = os.path.join(directory, filename)
  #     photo_filenames.append(path)

  # return photo_filenames, sorted(class_names)

  # labels, associate with a specific file
  labels = scipy.io.loadmat(os.path.join(
      dataset_dir, 'labels.mat'))
  labels = labels['labels'][0]
  assert 0 not in labels
  labels -= 1  # start from 0

  file_names = []
  images_root = os.path.join(dataset_dir, 'jpg')
  for file_name in os.listdir(images_root):
    path = os.path.join(images_root, file_name)
    file_names.append(path)

  file_names = sorted(file_names)
  file_names_to_labels = {file_names[i]: labels[i] for i in range(len(labels))}

  return file_names, file_names_to_labels


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'flowers102_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, file_names_to_labels, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' %
                             (i + 1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = CLASS_NAMES[file_names_to_labels[filenames[i]]]
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(image_data, b'jpg',
                                                       height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, 'jpg')
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  # if _dataset_exists(dataset_dir):
  #   print('Dataset files already exist. Exiting without re-creating them.')
  #   return

  # Download stuffs
  dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  _download_file(_LABELS_URL, os.path.join(dataset_dir, 'labels.mat'))
  _download_file(_SPLIT_URL, os.path.join(dataset_dir, 'split.mat'))

  file_names, file_names_to_labels = _get_filenames_and_classes(dataset_dir)
  class_names = CLASS_NAMES
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  for i in range(10):
    label = file_names_to_labels[file_names[i]]
    print(file_names[i], label, class_names[label])

  # Divide into train and test:
  # random.seed(_RANDOM_SEED)
  # random.shuffle(photo_filenames)

  split = scipy.io.loadmat(os.path.join(dataset_dir, 'split.mat'))
  trnid = list(split['trnid'][0])
  valid = list(split['valid'][0])
  tstid = list(split['tstid'][0])

  train_file_names = []
  valid_file_names = []
  for i in trnid:
    train_file_names.append(file_names[i-1])
  for i in valid:
    train_file_names.append(file_names[i-1])
  for i in tstid:
    valid_file_names.append(file_names[i-1])

  # First, convert the training and validation sets.
  _convert_dataset('train', train_file_names, class_names_to_ids,
                   file_names_to_labels, dataset_dir)
  _convert_dataset('validation', valid_file_names, class_names_to_ids,
                   file_names_to_labels, dataset_dir)

  # # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  # _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the Flowers-102 dataset!')
