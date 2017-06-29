import tensorflow as tf
import numpy as np
import os

import Input
import Process

def restore_model():
    with tf.Session() as sess:
         with tf.device('/cpu:0'):
             images, labels = Process.eval_inputs()
             #images, labels = Process.inputs()
             nn_input = Process.forward_propagation(images, 1.0)
             softmax = tf.nn.softmax(nn_input)
             results = tf.nn.top_k(nn_input, k=1, sorted=False, name=None)
             saver = tf.train.Saver()
             saver.restore(sess, 'nn-data/model')
             print("ConvNet Loaded Successfully")
             tf.train.start_queue_runners(sess = sess)
             for i in xrange(16):
                 #print(sess.run([results, labels]))
		         print(sess.run([results, labels]))

def main(argv = None):
    with tf.device('/cpu:0'):
    	restore_model()

if __name__ == '__main__':
    tf.app.run()
