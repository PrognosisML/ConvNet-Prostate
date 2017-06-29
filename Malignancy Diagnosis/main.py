import Process

import time
import numpy as np
import os

import tensorflow as tf
from six.moves import xrange
from datetime import datetime

FLAGS = tf.app.flags.FLAGS


def train():
    with tf.Graph().as_default():

        images, labels = Process.inputs()

        tf.summary.image("Base_Image", images, max_outputs=2, collections=None)

        forward_propagation_results = Process.forward_propagation(images, 0.5)

        softmax_debug = tf.nn.softmax(forward_propagation_results)

        cost = Process.error(forward_propagation_results, labels)

        train_op = Process.train(cost)

        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        sess.run(init)

        saver = tf.train.Saver()

        tf.train.start_queue_runners(sess = sess)

        train_dir = "nn-data"

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        for step in xrange(520000):
            start_time = time.time()
            loss_value, _, debug, labelsa = sess.run([cost, train_op, softmax_debug, labels])
            print(debug)
            duration = time.time() - start_time
            #accuracy = sess.run(train_accuracy)

            if step % 1 == 0:
                examples_per_sec = 1 / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, (%.3f examples/sec; %.3f ''sec/batch) loss = %.3f')
                print (format_str % (datetime.now(), step, examples_per_sec, sec_per_batch, loss_value))
                print(labelsa)
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                checkpoint_path = os.path.join(train_dir, 'model')
                saver.save(sess, checkpoint_path)

def main(argv = None):
    train()

if __name__ == '__main__':
  tf.app.run()
