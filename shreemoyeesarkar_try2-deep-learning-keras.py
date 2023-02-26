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




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import Dropout, Dense, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))




train = pd.read_csv("../input/train.csv")

test_images = (pd.read_csv("../input/test.csv").values).astype('float32')




train_images = (train.ix[:,1:].values).astype('float32')
train_labels = train.ix[:,0].values.astype('int32')




train_images = train_images.reshape((42000, 28 * 28))




train_images = train_images / 255
test_images = test_images / 255




train_labels = to_categorical(train_labels)
num_classes = train_labels.shape[1]
num_classes




# DESIGNING NEURAL NETWORK




seed=43
np.random.seed(seed)




model=Sequential()
model.add(Dense(32,activation='relu',input_dim=(28 * 28)))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))




model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])




history=model.fit(train_images, train_labels, validation_split = 0.05, nb_epoch=15, batch_size=64)




history_dict = history.history
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()




predictions = model.predict_classes(test_images, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)






