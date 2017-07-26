
# coding: utf-8

# In[1]:

from __future__ import division, print_function, absolute_import 


# In[2]:

import tensorflow as tf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')


# In[20]:

df = pd.read_csv('mergeddata_june22.csv')
print(df.as_matrix().shape)
df


# In[4]:

prices = list(df.as_matrix()[:, 0])
sizes = list(df.as_matrix()[:, 3])
comunities = list(df.as_matrix()[:,9])


# In[5]:

print(prices[:10])


# In[6]:

# matrix_df = df.as_matrix()
# maxi = -1
# for i in range(matrix_df.shape[0]):
# #     if matrix_df[i][1] > maxi:
# #         maxi = matrix_df[i][1]
# #     print(matrix_df[i][2])
#     if matrix_df[i][1] == nan:
#         print("kk")
# # print(maxi)

# # matrix_df = matrix_df/np.amax(matrix_df, axis=0)

# # print(matrix_df.shape)


# In[7]:

def get_data(path):
    df = pd.read_csv(path)
    matrix_df = df.as_matrix()
#     mat = matrix_df[:, 1:]
    mat = matrix_df
    where_are_NaNs = np.isnan(mat)
    mat[where_are_NaNs] = 0
    maximums = np.amax(mat, axis=0)
#     print(maximums)
    mat = mat/maximums
    return mat 


# In[8]:

mat = get_data("mergeddata_june22.csv")
# print(mat[:5])


# In[25]:

print(mat[:3])


# In[9]:

def next_batch(mat, index, size):
    if index + size <= mat.shape[0]:
        return index+size, mat[index:index+size]
    else:
        return index+size-mat.shape[0], np.concatenate((mat[index:], mat[:index+size-mat.shape[0]]), axis=0)


# In[10]:

learning_rate = 0.01
training_epochs = 500 
batch_size = 50
display_step = 25
examples_to_show = 10
index = 0


# In[11]:

n_hidden_1 = 12
n_hidden_2 = 6
n_hidden_3 = 1
n_input = 22


# In[12]:

X = tf.placeholder("float", [None, n_input])


# In[13]:

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b3': tf.Variable(tf.random_normal([n_input]))
}


# In[22]:

health_weights = weights['encoder_h1'][7]
print(health_weights.shape)
print(type(health_weights))


# In[14]:

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    return layer_3

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    return layer_3


# In[15]:

# Construct model 
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)


# In[16]:

# Prediction 
y_pred = decoder_op 
y_true = X


# In[24]:

health_penalty = 10


# In[225]:

# loss and optimizer 
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2)) + 
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


# In[226]:

init = tf.global_variables_initializer()


# In[227]:

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

plt.plot(total_c)


# In[266]:

x_line = range(1,mat.shape[0]+1)
plt.figure(figsize=(50, 20))
plt.plot(x_line[1000:1100], scores[1000:1100], x_line[1000:1100], [com/100 for com in comunities][1000:1100])# [price/1000000 for price in prices][:2000], 'r--', x_line[:2000], [size/1000 for size in sizes][:2000], 'black')


# In[231]:

print(scores)


# In[233]:

print(scores[0])

