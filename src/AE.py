from __future__ import division, print_function, absolute_import 

import pickle 
import numpy as np
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt

from utility import get_data, next_batch, evaluate_by_std


class Autoencoder(object):
    '''
    '''
    def __init__(self, path="mergeddata_july.csv", 
                    bed_penalty=2.621724506, 
                    bath_penalty=4.563851847, 
                    size_penalty=2.205301921, 
                    neigh_penalty=15):
        self.learning_rate = 0.0001
        self.epochs = 600
        self.batch_size = 50
        self.display_step = 100
        self.examples_to_show = 10

        self.index = 0
        self.n_hidden_1 = 8
        self.n_hidden_2 = 4
        self.n_hidden_3 = 1
        self.n_input = 4

        self.bed_penalty = bed_penalty
        self.bath_penalty = bath_penalty
        self.size_penalty = size_penalty
        self.neigh_penalty = neigh_penalty

        self.features_list = ['bedrooms', 'bathrooms', 'size', 'neighborhood', 'price']
        self.features_dict = {}
        self.load_data(path)

    def load_data(self, path):
        try:
            self.mat = get_data(path)
            for idx, feature in enumerate(self.features_list):
                self.features_dict[feature] = np.squeeze(self.mat[:, idx])
            self.mat = self.mat[:, :-1]
            print("data loaded successfully")
        except:
            print("error when loading data")

    def set_epochs(self, epoch):
        self.epochs = epoch 

    def initialize_weights(self):
        self.X = tf.placeholder("float", [None, self.n_input])
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h3': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input]))
        }

        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.n_hidden_3])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([self.n_input]))
        }

        self.bed_weights = self.weights['encoder_h1'][0]
        self.bath_weights = self.weights['encoder_h1'][1]
        self.size_weights = self.weights['encoder_h1'][2]
        self.neigh_weights = self.weights['encoder_h1'][3]

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
        self.initialize_weights()
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op) 
        self.y_pred = self.decoder_op 
        self.y_true = self.X

        self.cost = 10000 * tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))  \
                    + 0.07 * (self.bed_penalty * tf.reduce_sum(tf.abs(self.bed_weights))  \
                    + self.bath_penalty * tf.reduce_sum(tf.abs(self.bath_weights))  \
                    + self.neigh_penalty * tf.reduce_sum(tf.abs(self.neigh_weights))  \
                    + self.size_penalty * tf.reduce_sum(tf.abs(self.size_weights)))

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
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: batch_xs})
            
            total_c.append(c)
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),"cost=", "{:.9f}".format(c))
        print("Optimization Finished!")

    def predict(self, input_data=None):
        if input_data == None:
            input_data = self.mat
        
        score = self.sess.run(self.encoder_op, feed_dict={self.X: input_data})
        # print(np.squeeze(score).shape)
        return np.squeeze(score)

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

    def evaluate(self, scores, comp_sequences=None, sample_range=(100, 600)):
        # get comparison plots 
        f, axarr = plt.subplots(5, sharex=True, figsize=(100, 50))
        x_line = range(1,scores.shape[0]+1)
        # plt.figure(figsize=(100, 50))
        for idx, feature in enumerate(self.features_list):
            axarr[idx].plot(x_line[sample_range[0]:sample_range[1]], scores[sample_range[0]:sample_range[1]], \
             x_line[sample_range[0]:sample_range[1]], self.features_dict[feature][sample_range[0]:sample_range[1]], 'k')
            axarr[idx].set_title('scores v.s. '+feature)
        plt.savefig('plots/result' + str(self.neigh_penalty) + '.png')

        # get quantitative criterion
        criterion = 0
        if comp_sequences == None:
            for key in self.features_dict:
                criterion += evaluate_by_std(scores, self.features_dict[key])
        else:
            for seq in comp_sequences:
                criterion += evaluate_by_std(scores, seq)
        return criterion
        
    
class NeighborhoodAE(Autoencoder):
     def __init__(self):
         pass 


def test_AE(neigh_penalty=15.11173694):
    AE = Autoencoder(path='mergeddata_july.csv', neigh_penalty=neigh_penalty)
    AE.build()
    AE.train() 
    scores = AE.predict()
    AE.save()
    print(scores[100:600])
    criterion = AE.evaluate(scores=scores)
    AE.close_sess()
    print("the std is", criterion)
    return criterion


if __name__ == "__main__":
    """ test the autoencoders 
    """
    test_AE(neigh_penalty=15.11173694)
    # test_AE(neigh_penalty=15)
    # test_AE(neigh_penalty=20)

