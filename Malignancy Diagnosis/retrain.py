import tensorflow as tf
import numpy as np
import os

import Input
import Process
import time

from six.moves import xrange
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

def restore_model():
    with tf.Session() as sess:
        with tf.Graph().as_default() as g:
            with tf.device('/cpu:0'):
                images, labels = Process.inputs()
                nn_input = Process.forward_propagation(images, 0.5)
                softmax = tf.nn.softmax(nn_input)
                cost = Process.error(nn_input, labels)
                train_op = Process.train(cost)

                summary_op = tf.summary.merge_all()
                init = tf.global_variables_initializer()
                sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                sess.run(init)

                saver = tf.train.Saver()
                saver.restore(sess, 'nn-data/model')

                summary_writer = tf.summary.FileWriter("nn-data/model", sess.graph)

                print("ConvNet Loaded Successfully")
                tf.train.start_queue_runners(sess = sess)
                for step in xrange(10000):
                    start_time = time.time()
                    loss_value, _, debug, labelsa = sess.run([cost, train_op, softmax, labels])
                    print(debug)
                    duration = time.time() - start_time
                    if step % 1 == 0:
                        examples_per_sec = 1 / duration
                        sec_per_batch = float(duration)
                        format_str = ('%s: step %d, (%.3f examples/sec; %.3f ''sec/batch) loss = %.3f')
                        print (format_str % (datetime.now(), step, examples_per_sec, sec_per_batch, loss_value))
                        print(labelsa)
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                        saver.save(sess, 'nn-data/model')


def main(argv = None):
    restore_model()

if __name__ == '__main__':
    tf.app.run()
