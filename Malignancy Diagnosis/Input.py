from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob, os

from six.moves import xrange


NUM_CLASSES = 4

"""
This is a model based on the CIFAR10 Model.
The general structure of the program and a few functions are
borrowed from Tensorflow example of the CIFAR10 model.

https://github.com/tensorflow/tensorflow/tree/r0.7/tensorflow/models/image/cifar10/

As quoted:

"If you are now interested in developing and training your own image classification
system, we recommend forking this tutorial and replacing components to address your
 image classification problem."

Source:
https://www.tensorflow.org/tutorials/deep_cnn/

"""

def read_data(filename_queue):
  class Record(object):
    pass
  result = Record()

  label_bytes = 1
  result.height = 200
  result.width = 200
  result.depth = 3

  image_bytes = result.height * result.width * result.depth

  record_bytes = label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)

  result.key, value = reader.read(filename_queue)

  record_bytes = tf.decode_raw(value, tf.uint8)

  result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]), [result.depth, result.height, result.width])

  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def _generate_image_and_label_batch(image, label, batch_size):
  num_preprocess_threads = 8

  images, label_batch = tf.train.batch([image, label], batch_size = batch_size, capacity=batch_size)

  return images, tf.reshape(label_batch, [batch_size])

def eval_generate_image_and_label_batch(image, label, batch_size):
  num_preprocess_threads = 8

  images, label_batch = tf.train.batch([image, label], batch_size = 1, capacity=1)

  return images, tf.reshape(label_batch, [batch_size])


def inputs(batch_size):
    filename = [os.path.join('General-Data', 'Prostate_Cancer_Data')]

    print(filename)

    filename_queue = tf.train.string_input_producer(filename)

    read_input = read_data(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    reshaped_image = tf.image.random_flip_left_right(reshaped_image)
    reshaped_image = tf.image.per_image_standardization(reshaped_image)

    height = 200
    width = 200

    return _generate_image_and_label_batch(reshaped_image, read_input.label, batch_size)

def eval_inputs(data_dir, batch_size):

  data_dir2 = 'VALIDATION'

  filenames = [os.path.join(data_dir2, 'Prostate_Cancer_Data1')]
  print(filenames)

  filename_queue = tf.train.string_input_producer(filenames)

  read_input = read_data(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  reshaped_image = tf.image.per_image_standardization(reshaped_image)
  height = 200
  width = 200

  return eval_generate_image_and_label_batch(reshaped_image, read_input.label, batch_size)
