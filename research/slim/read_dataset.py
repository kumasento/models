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
tf.app.flags.DEFINE_boolean(
    'dump_image', False, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('num_samples', 10, 'Number of test samples')

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

    for i in range(FLAGS.num_samples):
      if i % 100 == 0:
        print("Checking step {:5d} ...".format(i))

      np_image, np_label = sess.run([image, label])
      if len(np_image.shape) != 3 or FLAGS.dump_image:
        np_image = np_image[0] if len(np_image.shape) != 3 else np_image
        fig, ax = plt.subplots()
        ax.imshow(np_image)
        ax.set_title(dataset.labels_to_names[np_label])
        fig.savefig(os.path.join(
          FLAGS.dumpdir, '{}_example_{:03d}.pdf'.format(FLAGS.dataset_name, i)))
        plt.close(fig)

    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run(main)

