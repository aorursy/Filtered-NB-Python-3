#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'Greys'
plt.rcParams['image.interpolation'] = 'nearest'




IMAGE_WIDTH = IMAGE_HEIGHT = 28
IMAGE_SIZE = IMAGE_WIDTH*IMAGE_HEIGHT
LABELS = 10




data_train = pd.read_csv('../input/train.csv')




labels = np.array(data_train.pop('label')) # Remove the labels as a numpy array from the dataframe
labels = np.array([np.arange(LABELS) == label for label in labels])
data = np.array(data_train, dtype=np.float32)/255.0




plt.plot(np.argmax(labels[0:200], axis=1))




def showImage(image_data):
    plt.imshow(image_data.reshape(IMAGE_WIDTH, IMAGE_HEIGHT))
    plt.axis('off')
    plt.show()
    
for i in range(1,4):
    showImage(data[[i]])




x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
W = tf.Variable(tf.random_uniform([IMAGE_SIZE, LABELS]))
b = tf.Variable(tf.random_uniform([LABELS]))

# 
y_ = tf.placeholder(tf.float32, [None, 10])

# Prediction
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Evaluate model
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)




sess = tf.InteractiveSession()




sess.run(tf.global_variables_initializer())
evolution = []
for i in range(10000):
    row_selection = np.random.permutation(40000)[0:100]
    train_step.run(feed_dict={x: data[row_selection], y_: labels[row_selection]})
    evolution += [cross_entropy.eval(feed_dict={x: data[row_selection], y_: labels[row_selection]})]
    
plt.plot(evolution)




correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: data[40000:], y_: labels[40000:]}))




predictions = tf.argmax(y,1).eval(session=sess, feed_dict={x:data[100:200]})
for i in range(3):
    showImage(data[100+i])
    print( "Prediction of img #{0} is {1}".format(100+i, predictions[i]))




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

x_image = tf.reshape(x, [-1,28,28,1])
conv = tf.nn.conv2d(x_image, weight_variable([5, 5, 1, 32]), strides=[1, 1, 1, 1], padding='SAME')
In [12]:
conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
conv.eval(feed_dict={x:data[0]})




#conv.eval(feed_dict={x: data[0]})
x_image.eval(feed_dict={x:data[0]})

