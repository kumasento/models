#!/usr/bin/env python
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from datasets import dataset_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('dumpdir', None, 'Where to dump output')
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

  batch_size = 16
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=1,
      common_queue_capacity=20 * batch_size,
      common_queue_min=10 * batch_size)
  [image, label] = provider.get(['image', 'label'])

  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    for i in range(10):
      np_image, np_label = sess.run([image, label])

      fig, ax = plt.subplots()
      ax.imshow(np_image)
      ax.set_title(dataset.labels_to_names[np_label])
      fig.savefig(os.path.join(FLAGS.dumpdir, 'flowers102_example_{:03d}.pdf'.format(i)))
      plt.close(fig)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run(main)

