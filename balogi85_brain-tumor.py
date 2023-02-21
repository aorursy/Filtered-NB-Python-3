#!/usr/bin/env python
# coding: utf-8



import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from numpy import expand_dims
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adadelta, Adamax

get_ipython().run_line_magic('matplotlib', 'inline')




path = '../input/brain-mri-images-for-brain-tumor-detection/'




list_dir = os.listdir(path)




print(list_dir)




yes = cv2.imread('../input/brain-mri-images-for-brain-tumor-detection/yes/Y1.jpg')
no = cv2.imread('../input/brain-mri-images-for-brain-tumor-detection/no/19 no.jpg')




plt.imshow(yes)




plt.imshow(no)




name_path = []
shape1=[]
shape2=[]
shape3=[]

for i in os.listdir(path+'yes/'):
    item = cv2.imread(path + 'yes/' + i)
    item.shape
    name_path.append(path + 'yes/' + i)
    shape1.append(item.shape[0])
    shape2.append(item.shape[1])
    shape3.append(item.shape[2])




train_df = pd.DataFrame(columns=['name', 'width', 'height', 'ch', 'class'])
train_df['name'], train_df['width'], train_df['height'], train_df['ch'], train_df['class'] = name_path, shape1, shape2, shape3, 'yes'
train_df.tail()




name_path = []
shape1=[]
shape2=[]
shape3=[]

for i in os.listdir(path+'no/'):
    item = cv2.imread(path + 'no/' + i)
    item.shape
    name_path.append(path + 'no/' + i)
    shape1.append(item.shape[0])
    shape2.append(item.shape[1])
    shape3.append(item.shape[2])




no_df = pd.DataFrame(columns=['name', 'width', 'height', 'ch', 'class'])
no_df['name'], no_df['width'], no_df['height'], no_df['ch'], no_df['class'] = name_path, shape1, shape2, shape3, 'no'
no_df.head()




no_df.width.unique()




no_df.height.unique()




no_df.ch.unique()




print(no_df.width.min(), no_df.height.min())




test_df = train_df.iloc[-10:-1, :]




test_df




train_df.drop(train_df.index[145:154], inplace=True)




len(train_df.index)




test_df_no = no_df.iloc[-10:-1, :]




test_df_no




no_df.drop(no_df.index[88:97], inplace=True)




len(no_df.index)




test = test_df.append(test_df_no, ignore_index = True) 




len(test.index)




train = train_df.append(no_df, ignore_index = True) 




len(train.index)




train.head()




test.head()




shape = 112
batch_size = 32
learn_r = 0.001




train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=30, width_shift_range=0.2, 
                                   height_shift_range=0.2, zoom_range=0.1, vertical_flip=True, horizontal_flip=True, validation_split=0.2)




train_generator = train_datagen.flow_from_dataframe(train, target_size=(shape, shape),batch_size=batch_size,class_mode='binary',subset='training', 
                                                   x_col='name',y_col="class", color_mode="rgb", shuffle=True)

validation_generator = train_datagen.flow_from_dataframe(train, target_size=(shape, shape),batch_size=batch_size,class_mode='binary',subset='validation', 
                                                   x_col='name',y_col="class", color_mode="rgb", shuffle=True)




img = load_img('../input/brain-mri-images-for-brain-tumor-detection/yes/Y107.jpg')

data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(height_shift_range=0.2)
it = datagen.flow(samples, batch_size=1)
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	batch = it.next()
	image = batch[0].astype('uint8')
	pyplot.imshow(image)
pyplot.show()




from tensorflow.keras.applications import VGG16, VGG19, EfficientNetB7, Xception, InceptionV3
opt_1 = SGD(learning_rate=learn_r, momentum=0.9)
opt_2 = Adam(learning_rate= learn_r)
opt_3 = Nadam(learning_rate= learn_r)
opt_4 = Adamax(learning_rate= learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-07)




apps = [VGG16, VGG19, EfficientNetB7, Xception, InceptionV3]




def get_model():    
    base_model =  apps[4](input_shape=(shape,shape,3), weights='imagenet', include_top=False, pooling='avg')
    x = base_model.output
    x = Dropout(0.1)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base_model.input, outputs=predictions)




model = get_model()
    
model.compile(optimizer=opt_1, loss='binary_crossentropy', metrics= 'accuracy')




history = model.fit_generator(train_generator, epochs=30, validation_data=validation_generator)




pd.DataFrame(history.history).plot(figsize=(18, 5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()




test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_dataframe(test, x_col='name', y_col='class', batch_size= 1, shuffle=False, target_size=(shape,shape))




test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1)




test_generator.class_indices




test_generator.filenames




predict = []
for i in pred:
    if i < 0.5:
        print('no')
        predict.append('no')
    else:
        print('yes')
        predict.append('yes')
    




results = pd.DataFrame(columns=['Filename', 'Predictions'])




results.head()




results['Filename']= test_generator.filenames




results['Predictions']=predict




for pic, name in zip(results['Filename'], results['Predictions']): 
    img = load_img(pic)
    plt.imshow(img)
    print(name)   
    plt.show()






