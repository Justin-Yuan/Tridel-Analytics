from __future__ import division, print_function, absolute_import 

import pickle 
import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt

from utility import get_data, next_batch


class Autoencoder(object):
    '''
    '''
    def __init__(self, path="autoencoder.csv"):
        self.learning_rate = 0.0001
        self.epochs = 800
        self.batch_size = 50
        self.display_step = 100
        self.examples_to_show = 10

        self.index = 0
        self.n_hidden_1 = 12
        self.n_hidden_2 = 6
        self.n_hidden_3 = 1
        self.n_input = 22

        self.load_data(path)

    def load_data(self, path):
        try:
            self.mat = get_data(path)
        except:
            print("error when loading data")

    def set_epochs(self, epoch):
        self.epochs = epoch 

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
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['encoder_h3']), self.biases['encoder_b3']))
        return layer_3

    def decoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['decoder_h3']), self.biases['decoder_b3']))
        return layer_3

    def build(self, path="autoencoder.csv"):
        # Construct model 
        self.encoder_op = encoder(self.X)
        self.decoder_op = decoder(self.encoder_op) 
        self.y_pred = self.decoder_op 
        self.y_true = self.X

        self.cost = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2)) + 
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)
        self.init = tf.global_variables_initializer()  

        self.sess = tf.Session() 
        self.sess.run(self.init) 
        self.saver = tf.train.Saver()   

    def train(self, overwrite=False):
        total_c = []
        total_batch = int(self.mat.shape[0]/self.batch_size)
        
        if overwrite:
            self.sess.run(self.init)

        for epoch in range(self.epochs):
            for i in range(total_batch):
                self.index, batch_xs = next_batch(self.mat, self.index, self.batch_size)
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={X: batch_xs})
            
            total_c.append(c)
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
        print("Optimization Finished!")

    def predict(self, input_data=None):
        if input_data == None:
            input_data = self.mat
        
        score = self.sess.run(self.encoder_op, feed_dict={X: input_data)})
        scores = list(score)
        return scores 

    def save(self, path="model/model.ckpt"):
        save_path = self.saver.save(self.sess, path)
        print("model saved")

    def load(self, path="model/model.ckpt", sess=None):
        if sess == None:
            sess = self.sess
        self.saver.restore(sess, path)
        print("Model restored.")

    def close_sess(self):
        self.sess.close()
        print("closed session")
    
class NeighborhoodAE(Autoencoder):
     


if __name__ == "__main__":
    """ test the autoencoders 
    """

