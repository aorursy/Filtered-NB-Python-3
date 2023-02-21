#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import OrderedDict




train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')




x_train = train_df.iloc[:,1:].values
t_train_value = train_df.iloc[:,0].values
x_test = test_df.values




t_train = np.zeros((t_train_value.shape[0], 10))
for i in range(t_train_value.shape[0]):
    t_train[i, t_train_value[i]] = 1




class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
       
    def forward(self, x):
        self.x = x
        y = np.dot(x, self.W) + self.b
        return y
    
    def backward(self, dout):
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T)
        return dx




class Relu:
    def __init__(self):
        pass
    
    def forward(self, x):
        self.mask = (x < 0)
        return np.maximum(0, x)
    
    def backward(self, dout):
        dout[self.mask] = 0
        return dout




def softmax(x):
    x = x.T
    x = x - np.max(x)
    y = np.exp(x) / np.max(np.exp(x), axis=0)
    return y.T




def cross_entropy_error(y, t):
    return np.sum(t * np.log(y + 1e-7)) / y.shape[0]




class SoftmaxWithLoss:
    def __init__(self):
        pass
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout=1):
        return dout * (self.y - self.t) / self.y.shape[0]




class SGD:
    def __init__(self, lr=0.01):
        pass
    
    def update(self, param, )

