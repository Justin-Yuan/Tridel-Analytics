from __future__ import division, print_function, absolute_import 

import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt

from utility import get_data, next_batch


class Autoencoder(object):
    '''
    '''
    def __init__(self):
        self.learning_rate = 0.0001
        self.training_epochs = 800
        self.batch_size = 50
        self.display_step = 100
        self.examples_to_show = 10
        self.index = 0
        self.n_hidden_1 = 12
        self.n_hidden_2 = 6
        self.n_hidden_3 = 1
        self.n_input = 22

    def initialize_weights(self):
        self.X = tf.placeholder("float", [None, n_input])
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
            'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
        }

        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([n_input]))
        }

    def encoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
        return layer_3

    def decoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
        return layer_3

    def build(self):
        # Construct model 
        encoder_op = encoder(self.X)
        decoder_op = decoder(encoder_op) 
        y_pred = decoder_op 
        y_true = self.X

        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + 
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

    def train(self):

        with tf.Session() as sess:
            sess.run(init)
            total_c = []
            total_batch = int(mat.shape[0]/batch_size)
            
            for epoch in range(training_epochs):
                for i in range(total_batch):
                    index, batch_xs = next_batch(mat, index, batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
                
                total_c.append(c)
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
            print("Optimization Finished!")
            
            score = sess.run(encoder_op, feed_dict={X: mat})
            scores = list(score)

    def predict(self):
        pass 

    def evaluate(self):
        pass  

    