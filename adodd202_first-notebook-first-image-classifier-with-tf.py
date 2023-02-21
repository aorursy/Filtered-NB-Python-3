#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
#sample_sub = pd.read_csv('../input/sample_submission.csv')




import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
sess = tf.InteractiveSession()




# Defining all of the x data values, y data values, and x test data values.
# We may also want to split the current x and y data into validation data and training data.

x_data = train_data.values   # 42000 x 785, 42000 samples, 784 pixels per sample, need to remove label
y_data = x_data[:,0]     # 42000 labels, later converted to one hot vectors
x_data = np.delete(x_data, [0], axis=1)
x_data = x_data.astype(np.float)

x_test_data = test_data.values
x_test_data = x_test_data.astype(np.float)

#One hot vectors for y labels.
y_data_OneHot = np.zeros((42000, 10))
y_data_OneHot[np.arange(42000), y_data] = 1

#Tensorflow Variable set up
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Splitting data here, x_data, y_data
X_train, X_val, y_train, y_val = train_test_split(x_data, y_data_OneHot, test_size=0.2)




W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y))

train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)

epochs = 30

for __ in range (epochs):
    if __%20 == 0:
        print ("epoch", __)
    if __ == 99:
        print ("Done")
    for _ in range (1000):
        temp = np.random.randint(X_train.shape[0], size=100)
        X_train_batch = X_train[temp, :]
        y_train_batch = y_train[temp, :]
        train_step.run(feed_dict={x: X_train_batch, y_: y_train_batch})




correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: X_val, y_: y_val}))




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = .1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], 
                          strides = [1,2,2,1], padding = 'SAME')




W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)




W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)




W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)




W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2




cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

BATCH_SIZE = 100

init = tf.global_variables_initializer()#
sess = tf.InteractiveSession()#

sess.run(init)#

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
for i in range (8000):
    #Make a batch
    temp = np.random.randint(X_train.shape[0], size=BATCH_SIZE)
    X_train_batch = X_train[temp, :]
    y_train_batch = y_train[temp, :]
        
    #Print the training accuracy
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: X_train_batch, y_: y_train_batch, keep_prob: 1.0})
        print('Step %d, training accuracy %g' % (i, train_accuracy))
        
    #Train
    train_step.run(feed_dict ={x: X_train_batch, y_: y_train_batch, keep_prob: .5})
    
print ('validation data accuracy %g' % accuracy.eval(feed_dict = {
x: X_val, y_: y_val, keep_prob: 1.0}))
    




# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y_conv,1)

#print (x_test_data[0])

#predicted_labels = predict.eval(feed_dict={x: x_test_data, keep_prob: 1.0})

predicted_labels = np.zeros(x_test_data.shape[0])
for i in range(0,x_test_data.shape[0]//BATCH_SIZE):
    if i%100 == 0:
        print(i)
        print (predict.eval(feed_dict
                    ={x: x_test_data[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0}))
        
    predicted_labels[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict
                            ={x: x_test_data[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0})
    #First batch computes 0-100, next batch is 101-200, etc
    
np.savetxt('submission_dodd.csv', 
       np.c_[range(1,len(x_test_data)+1),predicted_labels],
       delimiter=',', 
       header = 'ImageId,Label', 
       comments = '', 
       fmt='%d')                 

print (predicted_labels)
print (predicted_labels.shape)
print (np.mean(predicted_labels))
                              




print (np.mean(x_test_data))
sess.close()

