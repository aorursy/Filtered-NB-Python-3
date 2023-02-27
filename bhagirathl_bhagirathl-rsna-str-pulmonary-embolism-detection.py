#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




# https://www.kaggle.com/vbookshelf/cnn-how-to-use-160-000-images-without-crashing
from numpy.random import seed
seed(101)
#from tensorflow import set_random_seed
#set_random_seed(101)

import tensorflow as tf
tf.compat.v1.set_random_seed(101)


import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import os
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

SAMPLE_SIZE = 80000 # the number of images we use from each of the two classes




os.listdir('/kaggle/input')




print(len(os.listdir('/kaggle/input/rsna-str-pulmonary-embolism-detection/train')))
print(len(os.listdir('/kaggle/input/rsna-str-pulmonary-embolism-detection/test')))




#Create a Dataframe containing all images

df_data = pd.read_csv('/kaggle/input/rsna-str-pulmonary-embolism-detection/train.csv')

print(df_data.shape)




df_data.sample(2)




# add the path column with filenames in the dataframe

df_data['path'] = '/kaggle/input/rsna-str-pulmonary-embolism-detection/train/' + df_data['StudyInstanceUID'] + '/' + df_data['SeriesInstanceUID'] + '/' + df_data['SOPInstanceUID'] + '.dcm'




df_data.sample()




df_data.iloc[1621755,].path




import pydicom
from pydicom import dcmread
ds = pydicom.read_file('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/b4548bee81e8/ac1aea5d7662/cc96a7a2e72c.dcm')
ds = dcmread('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/b4548bee81e8/ac1aea5d7662/cc96a7a2e72c.dcm')
sample_image_arr = ds.pixel_array
sample_image_arr.shape
#show the sample image

import matplotlib.pyplot as plt
plt.imshow(sample_image_arr, cmap="gray")
plt.show()




sample_image_arr.shape




import pydicom
from pydicom import dcmread
ds = dcmread(df_data.iloc[1621756,].path)
sample_image_arr = ds.pixel_array
sample_image_arr.shape
#show the sample image

import matplotlib.pyplot as plt
plt.imshow(sample_image_arr, cmap="gray")
plt.show()




sample_image_arr.shape




# ImageDataGenerator cannot process .dcm file so we need to change it to .PNG
# initially we do it for a random sample of 1000 images and save them on /kaggle/tmp i.e. ../tmp
sample_size = 1000
df_data_1 = df_data.sample(sample_size)




df_data_1.describe()








# import numpy as np
# import png, os, pydicom

# # source_folder = r'path\to\source'
# # output_folder = r'path\to\output\folder'


# def dicom2png(source_folder, output_folder):
#     list_of_files = os.listdir(source_folder)
#     for file in list_of_files:
#         try:
#             ds = pydicom.dcmread(os.path.join(source_folder,file))
#             shape = ds.pixel_array.shape

#             # Convert to float to avoid overflow or underflow losses.
#             image_2d = ds.pixel_array.astype(float)

#             # Rescaling grey scale between 0-255
#             image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

#             # Convert to uint
#             image_2d_scaled = np.uint8(image_2d_scaled)

#             # Write the PNG file
#             with open(os.path.join(output_folder,file)+'.png' , 'wb') as png_file:
#                 w = png.Writer(shape[1], shape[0], greyscale=True)
#                 w.write(png_file, image_2d_scaled)
#         except:
#             print('Could not convert: ', file)


# cv2.imwrite(outdir + f.replace('.dcm','.png'),img)dicom2png(source_folder, output_folder)





# df_data['path_new'] = df_data['SOPInstanceUID'] + '.png'




# df_data['path_new']




# len(df_data)




# import cv2
# outdir = '/kaggle/tmp/'
# #os.mkdir(outdir)


# for i in range(len(df_data)):
#     ds = dcmread(df_data.iloc[i,].path)
#     image_arr = ds.pixel_array
#     image_arr.shape
#     cv2.imwrite(outdir + df_data.iloc[i,].SOPInstanceUID + '.png',image_arr)
#     print(i)
    
# #show the sample image





# get_ipython().system('ls /kaggle/tmp')




# df_data_1.path_new




# import os
# for dirname, _, filenames in os.walk('/kaggle/tmp'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))




# # Create a data generator
# datagen = ImageDataGenerator(rescale=1./255,validation_split=0.20)




# # load and iterate training dataset
# train_it = datagen.flow_from_dataframe(df_data,directory='/kaggle/tmp/',x_col = 'path_new',y_col = 'pe_present_on_image',class_mode = 'raw',batch_size = 64,validate_filenames=False, color_mode = 'grayscale',target_size = (512,512))
# # load and iterate validation dataset
# #val_it = datagen.flow_from_directory('/kaggle/input/rsna-str-pulmonary-embolism-detection/test/', class_mode='categorical', batch_size=64)
# # load and iterate test dataset
# #test_it = datagen.flow_from_directory('/kaggle/input/rsna-str-pulmonary-embolism-detection/test/', class_mode='categorical', batch_size=64)




# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
# from tensorflow.keras.utils import to_categorical, plot_model

# # network parameters
# input_shape = (512,512,1)
# batch_size = 128
# kernel_size = (3,3)
# pool_size = 2
# filters = 64
# dropout = 0.2
# num_labels = 2

# model = Sequential()
# model.add(Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu', input_shape = input_shape))
# model.add(MaxPooling2D(pool_size))
# model.add(Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu'))
# model.add(MaxPooling2D(pool_size))
# model.add(Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu'))
# model.add(Flatten())
# model.add(Dropout(dropout))
# model.add(Dense(num_labels))
# model.add(Activation('softmax'))
# model.summary()




# plot_model(model,to_file='cnn-mnist.png',show_shapes = True)




# model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])




# model.fit_generator(generator=train_it,
#                     steps_per_epoch=8,
#                     epochs=10)




# df_data.columns




# df_data.dtypes




# df_data['pe_present_on_image'].unique()




# df_data['pe_present_on_image'].isnull().count()






