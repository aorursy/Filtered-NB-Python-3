#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.np_utils import to_categorical
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

ax = sns.countplot(x="label", data=train, palette="Set3")
Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train.values.reshape(-1,28,28,1)




# normalization
X_train = X_train/255.




# plot the data
g = plt.imshow(X_train[3][:,:,0])
print(Y_train[3])




# Label encoding
Y_train = to_categorical(Y_train, num_classes = 10)
Y_train[3]




Y_train.shape




def create_placeholders(n_H, n_W, n_C, n_y):
    """
    Create placeholders which are required for tensorflow session.
    n_H - height of the image
    n_W - width of the image
    n_C - channels of image
    n_y - number of output classes
    
    Returns: X- placeholder for data input, shape [None, n_H, n_W, n_C]
             Y - placeholer for input labels, of shape [None, n_y]"""
    
    X = tf.placeholder(tf.float32, shape = (None, n_H, n_W, n_C))
    Y = tf.placeholder(tf.float32, shape = (None, n_y))
    
    return X, Y
    




X, Y = create_placeholders(X_train.shape[1],X_train.shape[2],X_train.shape[3],Y_train.shape[1])
print("X = " + str(X))
print("Y = " + str(Y))




def initialization

