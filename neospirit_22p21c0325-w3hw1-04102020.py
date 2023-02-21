#!/usr/bin/env python
# coding: utf-8



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




get_ipython().system('nvidia-smi')




import tensorflow as tf
import keras

import sklearn
from sklearn.metrics import jaccard_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import NASNetMobile, Xception, DenseNet121, MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime

import numpy as np
import os
import cv2
import pandas as pd
# import imutils
import random
from PIL import Image
import matplotlib.pyplot as plt




DATA_PATH = "../input/super-ai-image-classification/"

TRAIN_PATH = DATA_PATH + "train/train/"
TEST_PATH = DATA_PATH + "val/val/"




seed = 42

BATCH_SIZE = 15
IMAGE_SIZE = 224




df = pd.read_csv(TRAIN_PATH + "train.csv")




df




# Find amount of category "1"
print(sum(df["category"]))




df.index = df['id']
df.category["05362d55-c21d-49da-8ddf-4fab2e68f8fc.jpg"]




def load_img():
    label = []
    img_path = os.listdir(TRAIN_PATH + "images")
    train_img = np.zeros((len(img_path), 224, 224, 3)).astype('float')
    
    count = 0
    for image in img_path:
        train_img1 = cv2.imread(TRAIN_PATH + "images/"+image)/255.
        train_img1 = cv2.resize(train_img1, (224, 224))
        print(f"Image {count}: ",train_img1.shape,df.category[image])
        train_img[count] = train_img1
        label.append(df.category[image])
        print(f"Load Image {count}: Complete!")
        count += 1
        
    return train_img, label




train_img, label = load_img()




print("Train image amount",len(label),f"images.")
print(f"shape: {train_img.shape}")




plt.imshow(train_img[4].reshape(224,224,3))
plt.show()




val_img = train_img.copy()[:345]
val_mask = label[:345]

train_img = train_img.copy()[345:]
train_mask = label[345:]

train_datagen = ImageDataGenerator(
    rotation_range=5,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.15,
    fill_mode="nearest")

val_datagen = ImageDataGenerator()



# Keep the same seed for image and mask generators so they fit together
train_generator= train_datagen.flow(train_img, train_mask, batch_size=BATCH_SIZE, seed=seed)
val_generator  = val_datagen.flow(val_img, val_mask, batch_size=BATCH_SIZE, seed=seed)




# print(f"Step of train images, shape: {len(train_img_1)}, {train_img_1[1].shape}")
# print(f"Step of validation images, shape: {len(val_img_1)}, {train_img_1[1].shape}")




base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')




model.summary()




with tf.device('/device:GPU:0'):                # Initialize process to GPU
    def build_model():
        inputs = tf.keras.Input(shape=(224, 224, 3))
        base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3),
                                               include_top=False,
                                               weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False
        feature_batch = base_model(inputs)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        feature_batch_average = Dropout(0.2)(feature_batch_average)
        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average) 
        x = Activation('sigmoid')(prediction_batch)

        model = Model(inputs, x)
        return model
    
    import keras as K
    def dice_coef(y_true, y_pred, smooth=1):                                                                                # Dice coefficient using for validate predict image to truth mask.
        y_true_f = K.backend.flatten(y_true)
        y_pred_f = K.backend.flatten(y_pred)
        intersection = K.backend.abs(K.backend.sum(y_true * y_pred))
        union = K.backend.abs(K.backend.sum(y_true_f)) + K.backend.abs(K.backend.sum(y_pred_f))
        dice = K.backend.mean((2. * intersection + smooth)/(union + smooth))                                                  # Dice coefficient equation : Dice = 2*abs(intersection)/abs(union)   //smooth using for make model learning easier
        return dice

    def dice_coef_loss(y_true, y_pred):                                                                                     # Using dice coeffiecient as a loss function                          // Loss is alike to error of the model
        return 1 - dice_coef(y_true, y_pred)

    model = build_model()
    model.compile(optimizer="sgd",loss='binary_crossentropy', metrics=['accuracy'])
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard("./logs/")
    earlystopper = EarlyStopping(patience=10, mode=max, monitor='val_accuracy', verbose=1)
    csv_logger = CSVLogger('straight-final-mobilenetv2.csv', append=True, separator=';')
    checkpointer = ModelCheckpoint('./superaiImageclassificcation_mobilenetv2.h5', verbose=1, save_best_only=True, mode='max', monitor='val_accuracy')

fitting = model.fit(train_generator,
                    epochs=100, validation_data=val_generator,callbacks= [csv_logger,earlystopper,checkpointer])




best_model = load_model("./superaiImageclassificcation_mobilenetv2.h5")
train_acc = best_model.evaluate(val_img,np.array(val_mask), verbose=0)
print(train_acc)




pred = model.predict(train_img)




label = []
for value in range(len(pred)):
    if pred[value] >= 0.5:
        label.append(1)
    else:
        label.append(0)




print(label[1:10])




df['fuck_bitch'] = label
df




def load_img_test():
    img_path = os.listdir(TEST_PATH + "images")
    test_img = np.zeros((len(img_path), 224, 224, 3)).astype('float')
    
    count = 0
    for image in img_path:
        test_img1 = cv2.imread(TEST_PATH + "images/"+image)/255.
        test_img1 = cv2.resize(test_img1, (224, 224))
        print(f"Image {count}: ",test_img1.shape)
        test_img[count] = test_img1
        print(f"Load Image {count}: Complete!")
        count += 1
        
    return test_img

test_data = pd.read_csv(TEST_PATH + "val.csv")




test_img = load_img_test()
test_img = np.array(test_img)
print(test_img.shape)
pred = best_model.predict(test_img)




label = []
for value in range(len(pred)):
    if pred[value] >= 0.5:
        label.append(1)
    else:
        label.append(0)




print(label)




data = {'id':os.listdir(TEST_PATH + "images"), 'category':label}
submission_df = pd.DataFrame(data)
submission_df
submission_df.to_csv('submission.csv',index=False)




model.save('./superaiclassification_submit_2.h5')
model.save_weights('./superaiclassification_submit_2.hdf5')




inputs = Input(shape=(224,224,3))
encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False, alpha=0.35)




encoder = encoder()
encoder.summary()




def encoder():
       inputs = Input(shape=[224,224,3], name='input_image')

       #Pretrained Encoder
       encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False,alpha=0.35)
       skip_connection_names = ['input_image','block_1_expand_relu','block_3_expand_relu','block_6_expand_relu']
   #     encoder.summary()
       encoder_output = encoder.get_layer('block_13_expand_relu').output

       f = [16,32,48,64]
       x = encoder_output

       for i in range(1,len(f)+1):
           x_skip = encoder.get_layer(skip_connection_names[-i]).output
           x = UpSampling2D((2,2))(x)
           x = Concatenate()([x,x_skip])

           x = Conv2D(f[-i], (3,3),padding='same')(x)
           x = BatchNormalization()(x)
           x = Activation('relu')(x)

           x = Conv2D(f[-i], (3,3),padding='same')(x)
           x = BatchNormalization()(x)
           x = Activation('relu')(x)

       x = Conv2D(3, (1,1),padding='same')(x)
       x = Activation('sigmoid')(x)

       model = Model(inputs, x)
       return model
   

def build_modelResNet50():
   encoder = encoder()
   base_model = ResNet50(input_shape=(224,224,3),
                        include_top=False,
                        weights='imagenet')
   
   for layer in base_model.layers:
       layer.trainable = False
       
   feature_batch = base_model(encoder)
   global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
   feature_batch_average = global_average_layer(feature_batch)
   feature_batch_average = Dropout(0.2)(feature_batch_average)
   prediction_layer = tf.keras.layers.Dense(1)
   prediction_batch = prediction_layer(feature_batch_average) 
   x = Activation('sigmoid')(prediction_batch)

   model = Model(inputs, x)
   return model




with tf.device('/device:GPU:0'):                # Initialize process to GPU
    def encoder():
        inputs = Input(shape=[224,224,3], name='input_image')

        #Pretrained Encoder
        encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False,alpha=0.35)
        skip_connection_names = ['input_image','block_1_expand_relu','block_3_expand_relu','block_6_expand_relu']
        #     encoder.summary()
        encoder_output = encoder.get_layer('block_13_expand_relu').output

        f = [16,32,48,64]
        x = encoder_output

        for i in range(1,len(f)+1):
            x_skip = encoder.get_layer(skip_connection_names[-i]).output
            x = UpSampling2D((2,2))(x)
            x = Concatenate()([x,x_skip])

            x = Conv2D(f[-i], (3,3),padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            x = Conv2D(f[-i], (3,3),padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        x = Conv2D(3, (1,1),padding='same')(x)
        x = Activation('sigmoid')(x)

        model = Model(inputs, x)
        return model
    

    def build_modelResNet50():
        encoder = encoder()
        base_model = ResNet50(input_shape=(224,224,3),
                             include_top=False,
                             weights='imagenet')

        for layer in base_model.layers:
            layer.trainable = False

        feature_batch = base_model(encoder)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        feature_batch_average = Dropout(0.2)(feature_batch_average)
        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average) 
        x = Activation('sigmoid')(prediction_batch)

        model = Model(inputs, x)
        return model
    
    import keras as K
    def dice_coef(y_true, y_pred, smooth=1):                                                                                # Dice coefficient using for validate predict image to truth mask.
        y_true_f = K.backend.flatten(y_true)
        y_pred_f = K.backend.flatten(y_pred)
        intersection = K.backend.abs(K.backend.sum(y_true * y_pred))
        union = K.backend.abs(K.backend.sum(y_true_f)) + K.backend.abs(K.backend.sum(y_pred_f))
        dice = K.backend.mean((2. * intersection + smooth)/(union + smooth))                                                  # Dice coefficient equation : Dice = 2*abs(intersection)/abs(union)   //smooth using for make model learning easier
        return dice

    def dice_coef_loss(y_true, y_pred):                                                                                     # Using dice coeffiecient as a loss function                          // Loss is alike to error of the model
        return 1 - dice_coef(y_true, y_pred)

    model = build_model()
    model.compile(optimizer="sgd",loss='binary_crossentropy', metrics=['accuracy'])
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard("./logs/")
    earlystopper = EarlyStopping(patience=10, mode=max, monitor='val_accuracy', verbose=1)
    csv_logger = CSVLogger('straight-final-mobilenetv2.csv', append=True, separator=';')
    checkpointer = ModelCheckpoint('./superaiImageclassificcation_mobilenetv2.h5', verbose=1, save_best_only=True, mode='max', monitor='val_accuracy')

fitting = model.fit(train_generator,
                    epochs=100, validation_data=val_generator,callbacks= [csv_logger,earlystopper,checkpointer])




def encoder():
    inputs = Input(shape=[224,224,3], name='input_image')

    #Pretrained Encoder
    encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False,alpha=0.35)
    skip_connection_names = ['input_image','block_1_expand_relu','block_3_expand_relu','block_6_expand_relu']
    #     encoder.summary()
    encoder_output = encoder.get_layer('block_13_expand_relu').output

    f = [16,32,48,64]
    x = encoder_output

    for i in range(1,len(f)+1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2,2))(x)
        x = Concatenate()([x,x_skip])

        x = Conv2D(f[-i], (3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(f[-i], (3,3),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Conv2D(3, (1,1),padding='same')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x)
    return model
    

def build_modelResNet50():
    encoder = encoder()
    base_model = ResNet50(input_shape=(224,224,3),
                         include_top=False,
                         weights='imagenet')
    
    for layer in base_model.layers:
        layer.trainable = False
        
    feature_batch = base_model(encoder)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    feature_batch_average = Dropout(0.2)(feature_batch_average)
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average) 
    x = Activation('sigmoid')(prediction_batch)

    model = Model(inputs, x)
    return model




def MobileNetv3_large(num_classes=1000,input_shape=(224,224,3)):
    x = Input(shape=input_shape)
    out = keras.layers.Conv2D(16,3,strides=2,padding='same',use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out)
    out = Activation(Hswish)(out)

    out = Bneck(out,3,16,16,16,keras.layers.ReLU(),1,False)
    out = Bneck(out,3,16,64,24,keras.layers.ReLU(),2,False)
    out = Bneck(out,3,24,72,24,keras.layers.ReLU(),1,False)
    out = Bneck(out,5,24,72,40,keras.layers.ReLU(),2,True)
    out = Bneck(out,5,40,120,40,keras.layers.ReLU(),1,True)
    out = Bneck(out,5,40,120,40,keras.layers.ReLU(),1,True)
    out = Bneck(out,3,40,240,80,Activation(Hswish),2,False)
    out = Bneck(out,3,80,200,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,184,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,184,80,Activation(Hswish),1,False)
    out = Bneck(out,3,80,480,112,Activation(Hswish),1,True)
    out = Bneck(out,3,112,672,112,Activation(Hswish),1,True)
    out = Bneck(out,5,112,672,160,Activation(Hswish),1,True)
    out = Bneck(out,5,160,672,160,Activation(Hswish),2,True)
    out = Bneck(out,5,160,960,160,Activation(Hswish),1,True)

    out = keras.layers.Conv2D(filters=960,kernel_size=3,strides=1,padding='same')(out)
    out = keras.layers.BatchNormalization()(out)
    out = Activation(Hswish)(out)
    out = keras.layers.GlobalAveragePooling2D()(out)
    out = keras.layers.Reshape((1,1,-1))(out)
    out = keras.layers.Conv2D(filters=1280,kernel_size=1,strides=1)(out)
    out = Activation(Hswish)(out)
    out = keras.layers.Conv2D(filters=num_classes,kernel_size=1,strides=1)(out)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Softmax()(out)
    model = Model(inputs=x,outputs=out)
    return model




def Hswish(x):
    return x * tf.nn.relu6(x + 3) / 6

def Bneck(x, kernel_size, input_size, expand_size, output_size, activation,stride,use_se):
    out = keras.layers.Conv2D(filters=expand_size,kernel_size=1,strides=1,use_bias=False)(x)
    out = keras.layers.BatchNormalization()(out)
    out = activation(out)

    out = keras.layers.DepthwiseConv2D(kernel_size=kernel_size,strides=stride,padding='same')(out)
    out = keras.layers.BatchNormalization()(out)
    if use_se:
        out = SE_block(out)

    out = keras.layers.Conv2D(filters=output_size, kernel_size=1,strides=1,padding='same')(out)
    out = keras.layers.BatchNormalization()(out)

    if stride==1 and input_size != output_size:
        short_cut = keras.layers.Conv2D(filters=output_size,kernel_size=1,strides=1,padding='same')(x)
        out = keras.layers.Add()([out,short_cut])
    return out

def SE_block(x,reduction=4):
    filters = x._keras_shape[-1]
    out = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(int(filters/reduction),activation='relu')(out)
    out = keras.layers.Dense(filters,activation='hard_sigmoid')(out)
    out = keras.layers.Reshape((1,1,-1))(out)
    out = keras.layers.multiply([x,out])
    return out




with tf.device('/device:GPU:0'):                # Initialize process to GPU
    model = MobileNetv3_large(num_classes=1000,input_shape=(224,224,3))
    model.compile(optimizer='sgd', loss=tf.keras.losses.BinaryCrossentropy())
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard("./logs/")
    earlystopper = EarlyStopping(patience=10, verbose=1)
    csv_logger = CSVLogger('straight-final-mobilenetv2.csv', append=True, separator=';')
    checkpointer = ModelCheckpoint('./straight_sidewalk_model_augmented_v1_early10_mobilenetv2.h5', verbose=1, save_best_only=True)

fitting = model.fit(train_generator,
                    epochs=1000,
                    steps_per_epoch=len(train_img),
                    validation_data=val_generator,
                    validation_steps=len(val_img),
                    callbacks= [tensorboard_callback, csv_logger, checkpointer])

