import os
import sys

import tensorflow as tf
import Input

import os, re

FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'

tf.app.flags.DEFINE_integer('batch_size', 1, "hello")

def _activation_summary(x):
    with tf.device('/cpu:0'):
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inputs():
  images, labels = Input.inputs(batch_size = FLAGS.batch_size)
  return images, labels


def eval_inputs():
  data_dir = 'VALIDATION'
  images, labels = Input.eval_inputs(data_dir = data_dir, batch_size = 1)
  return images, labels

def weight_variable(name, shape):
    with tf.device('/gpu:0'):
        initial = tf.random_normal(shape, stddev=0.035)
        var = tf.Variable(initial, name)
        return var

def bias_variable(shape):
    with tf.device('/cpu:0'):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

def conv(images, W):
    return tf.nn.conv2d(images, W, strides = [1, 1, 1, 1], padding = 'SAME')

def forward_propagation(images, dropout_value):
    with tf.variable_scope("conv1"):
        with tf.device('/gpu:0'):
            conv1_feature = weight_variable('conv1_feature', [11, 11, 3, 10])
            conv1_bias = bias_variable([10])
            image_matrix = tf.reshape(images, [-1, 200, 200, 3])
            conv1_result = tf.nn.relu(conv(image_matrix, conv1_feature) + conv1_bias)
            _activation_summary(conv1_result)

            with tf.device('/cpu:0'):
                kernel_transposed = tf.transpose(conv1_feature, [3, 0, 1, 2])
                tf.summary.image('conv1/filters', kernel_transposed, max_outputs=10)

            conv1_pool = tf.nn.max_pool(conv1_result, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.variable_scope("conv2"):
        conv2_feature = weight_variable('conv2_feature', [3, 3, 10, 20])
        conv2_bias = bias_variable([20])
        conv2_result = tf.nn.relu(conv(conv1_pool, conv2_feature) + conv2_bias)
        _activation_summary(conv2_result)

        conv2_pool = tf.nn.max_pool(conv2_result, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.variable_scope("conv3"):
        conv3_feature = weight_variable('conv3_feature', [3, 3, 20, 30])
        conv3_bias = bias_variable([30])
        conv3_result = tf.nn.relu(conv(conv2_pool, conv3_feature) + conv3_bias)
        _activation_summary(conv3_result)

        conv3_pool = tf.nn.max_pool(conv3_result, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.variable_scope("conv4"):
        conv4_feature = weight_variable('conv4_feature', [3, 3, 30, 30])
        conv4_bias = bias_variable([30])
        conv4_result = tf.nn.relu(conv(conv3_pool, conv4_feature) + conv4_bias)

    with tf.variable_scope("conv5"):
        conv5_feature = weight_variable('conv5_feature', [3, 3, 30, 15])
        conv5_bias = bias_variable([15])
        conv5_result = tf.nn.relu(conv(conv4_result, conv5_feature) + conv5_bias)

    with tf.variable_scope("fcl"):
        perceptron1_weight = weight_variable('perceptron1_weight', [25 * 25 * 15, 25 * 25 * 15])
        perceptron1_bias = bias_variable([25 * 25 * 15])
        flatten_dense_connect = tf.reshape(conv5_result, [-1, 25 * 25 * 15])
        compute_perceptron1_layer = tf.nn.relu(tf.matmul(flatten_dense_connect, perceptron1_weight) + perceptron1_bias)
        dropout = tf.nn.dropout(compute_perceptron1_layer, dropout_value)
        _activation_summary(compute_perceptron1_layer)

        perceptron2_weight = weight_variable('perceptron2_weight', [25 * 25 * 15, 4])
        perceptron2_bias = bias_variable([4])

        result1 = tf.matmul(dropout, perceptron2_weight) + perceptron2_bias
        _activation_summary(result1)
        return result1

def error(forward_propagation_results, labels):
    with tf.device('/cpu:0'):
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=forward_propagation_results, labels=labels)
        cost = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('loss', cost)
        total_loss = tf.add_n(tf.get_collection('loss'), name='total_loss')
        _activation_summary(total_loss)
        return total_loss

def train(cost):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.95, staircase=True)
    train_loss = tf.train.GradientDescentOptimizer(learning_rate = rate).minimize(cost, global_step=global_step)
    return train_loss
