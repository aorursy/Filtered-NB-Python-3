#!/usr/bin/env python
# coding: utf-8



train.describe()




train_labels[0:9]




import matplotlib.pyplot as plt
plt.hist(train_labels)
plt.title("Frequency Histogram of Numbers in Training Data")
plt.xlabel("Number Value")
plt.ylabel("Frequency")




train_images




from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
num_classes = train_labels.shape[1]
num_classes




train_labels[0]




seed=98
np.random.seed(seed)




train_images.shape




train_labels.shape









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





train = pd.read_csv("../input/train.csv")
test_images=pd.read_csv("../input/test.csv")

train.head(2)




train.describe()




import matplotlib.pyplot as plt

train_images=(train.ix[:,1:].values).astype('float32')
train_labels=(train.ix[:,0].values).astype('int32')




train_labels[0:9]




import matplotlib.pyplot as plt
plt.hist(train_labels)
plt.title("Frequency Histogram of Numbers in Training Data")
plt.xlabel("Number Value")
plt.ylabel("Frequency")




# This will give array of 10 rows and 25 columns
train_images[10:25]
train_images.shape




#train_images array has data in (42k,784) shape .Each row is 28*28 pixel values.
#We will transform this into 3D array where each row will contian 28 by 28 matrix of pixel values.




train_images=train_images.reshape(train_images.shape[0],28,28)

# Here each value of the 1st dimension would correspond to 28 by 28 2D matrices




train_images.shape




train_images[0]




train_labels[0]




train_labels.dtype




train_labels




for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))
    plt.title(train_labels[i])




test_images.shape




train_images[1]




# This train_images has a dimension of 42k *28*28. Each row of first dimension correspond to two dimension matrix




train_images=train_images/255
test_images=test_images/255




train_images




train_images[6]




test_images[4]





# **One Hot encoding of labels.**

# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension.

# For example, 3 would be [0,0,0,1,0,0,0,0,0,0].




from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
num_classes = train_labels.shape[1]
num_classes




train_labels[0]






