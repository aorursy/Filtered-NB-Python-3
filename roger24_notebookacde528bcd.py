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




import pandas as pd
import numpy as np

df_train = pd.read_csv('../input/train.csv')

# 0.985714.  fisrt turnning, target--0.995
# USE real test.csv

from keras.models import Sequential
from keras.layers import Dense, Activation,Convolution2D,MaxPooling2D,Dropout,Flatten
import h5py


# model = Sequential([
# Dense(32, input_dim=784),
# Activation('relu'),
# Dense(10),
# Activation('softmax'),
#])

# ref: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1,28,28)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))     #(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

a = df_train.values
x_b = a[:,1:]
y_b = a[:,0:1]

# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test =  train_test_split(x_b,y_b,test_size=0.1, random_state=0)


X_train = x_b.reshape(x_b.shape[0], 1, 28, 28)
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# print X_train.shape, y_train.shape, X_test.shape

from keras.utils import np_utils
# convert class vectors to binary class matrices, OR WILL GET AN ERROR!
Y_train = np_utils.to_categorical(y_b, 10)
# Y_test = np_utils.to_categorical(y_test, 10)

model.fit(X_train, Y_train, batch_size=128, nb_epoch=10, verbose=1)
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'


X_test = pd.read_csv('data/test.csv')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
for i in X_test[:5]:
    print model.predict(i)
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

