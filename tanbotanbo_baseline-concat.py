#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import datetime
import random
import glob
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import BatchNormalization,Activation,Dropout,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
get_ipython().run_line_magic('matplotlib', 'inline')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 乱数シード固定
seed_everything(2020)




batch_size = 32
size = 128




from tensorflow.python.client import device_lib
device_lib.list_local_devices()




local = False
if local:
    inputPath = '../data/'
else:
    inputPath = '../input/4th-datarobot-ai-academy-deep-learning/'
# 画像読み込み
image = cv2.imread(inputPath+ '/images/train_images/' + '1_bathroom.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
display(image.shape)
display(image[0][0])
# 画像を表示
plt.figure(figsize=(8,4))
plt.imshow(image)




# 画像のサイズ変更
image = cv2.resize(image,(256,256))
display(image.shape)
display(image[0][0])
# 画像を表示
plt.figure(figsize=(8,4))
plt.imshow(image)




train = pd.read_csv(inputPath + 'train.csv')
test = pd.read_csv(inputPath + 'test.csv')
display(train.shape)
display(train.head())




def load_images(df, inputPath, size, roomType):
    images = []
    for i in df['id']:
        basePath = os.path.sep.join([inputPath, "{}_{}*".format(i,roomType)])
        housePaths = sorted(list(glob.glob(basePath)))
        for housePath in housePaths:
            image = cv2.imread(housePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (size, size))
        images.append(image)
    return np.array(images) / 255.0


roomType = 'frontal'
train_images1 = load_images(train, inputPath+"images/train_images/", size, roomType)
test_images1 = load_images(test, inputPath+"images/test_images/", size, roomType)

roomType = 'bathroom'
train_images2 = load_images(train, inputPath+"images/train_images/", size, roomType)
test_images2 = load_images(test, inputPath+"images/test_images/", size, roomType)

roomType = 'bedroom'
train_images3 = load_images(train, inputPath+"images/train_images/", size, roomType)
test_images3 = load_images(test, inputPath+"images/test_images/", size, roomType)

roomType = 'kitchen'
train_images4 = load_images(train, inputPath+"images/train_images/", size, roomType)
test_images4 = load_images(test, inputPath+"images/test_images/", size, roomType)

display(train_images1.shape)
display(train_images1[0][0][0])




def custom_cnn(inputShape):

    
    
    cnn_model = Sequential()
    """
    演習:kernel_sizeを変更してみてください
    """    
    cnn_model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='he_normal', input_shape=inputShape))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.1))

    cnn_model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', 
                     activation='relu', kernel_initializer='he_normal'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(BatchNormalization())
    cnn_model.add(Dropout(0.1))
    
    x1_input = Input(shape=inputShape)
    x2_input = Input(shape=inputShape)
    x3_input = Input(shape=inputShape)
    x4_input = Input(shape=inputShape)
    
    x1 = cnn_model(x1_input)
    x2 = cnn_model(x2_input)
    x3 = cnn_model(x3_input)
    x4 = cnn_model(x4_input)

    x1 = GlobalAveragePooling2D()(x1)
    x2 = GlobalAveragePooling2D()(x2)
    x3 = GlobalAveragePooling2D()(x3)
    x4 = GlobalAveragePooling2D()(x4)
    
    #####
    x = Concatenate(axis=-1)([x1, x2, x3, x4])
    # let's add a fully-connected layer
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=[x1_input, x2_input, x3_input, x4_input], outputs=predictions)
    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 
    return model




from tensorflow.keras.applications import ResNet50

def resnet_finetuning(inputShape):
    backbone = ResNet50(weights='imagenet',
                    include_top=False,
                    input_shape=inputShape)

    #for layer in backbone.layers[:165]:
    #    layer.trainable = False
    #for layer in backbone.layers:
    #    print("{}: {}".format(layer, layer.trainable))
    # 学習させるlayerを指定する
    n_frozen_layers = 165
    for i in range(len(backbone.layers)):
        # 前から(n_frozen_layers)個のlayerのパラメーターは学習不可にする。
        # ただし、Batch Normalization layerは全て学習可のままにする。
        backbone.layers[i].trainable =             (i >= n_frozen_layers) or             isinstance(backbone.layers[i], BatchNormalization)
    
    
    x1_input = Input(shape=inputShape)
    x2_input = Input(shape=inputShape)
    x3_input = Input(shape=inputShape)
    x4_input = Input(shape=inputShape)
    
    x1 = backbone(x1_input)
    x2 = backbone(x2_input)
    x3 = backbone(x3_input)
    x4 = backbone(x4_input)

    x1 = GlobalAveragePooling2D()(x1)
    x2 = GlobalAveragePooling2D()(x2)
    x3 = GlobalAveragePooling2D()(x3)
    x4 = GlobalAveragePooling2D()(x4)

    x = Concatenate(axis=-1)([x1, x2, x3, x4])
    # let's add a fully-connected layer
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    predictions = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[x1_input, x2_input, x3_input, x4_input], outputs=predictions)

    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 
    # model.summary()
    return model









from tensorflow.keras.applications import VGG16

def vgg16_finetuning(inputShape):
    backbone = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=inputShape)

    """
    演習:Convolution Layerの重みを全部訓練してみてください！
    """    
    
    for layer in backbone.layers[:19]:
        layer.trainable = False
    for layer in backbone.layers:
        print("{}: {}".format(layer, layer.trainable))
    x1_input = Input(shape=inputShape)
    x2_input = Input(shape=inputShape)
    x3_input = Input(shape=inputShape)
    x4_input = Input(shape=inputShape)
    
    x1 = backbone(x1_input)
    x2 = backbone(x2_input)
    x3 = backbone(x3_input)
    x4 = backbone(x4_input)

    x1 = GlobalAveragePooling2D()(x1)
    x2 = GlobalAveragePooling2D()(x2)
    x3 = GlobalAveragePooling2D()(x3)
    x4 = GlobalAveragePooling2D()(x4)

    x = Concatenate(axis=-1)([x1, x2, x3, x4])
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(rate=0.2)(x)
    predictions  = Dense(1, activation='linear')(x)
    
    model = Model(inputs=[x1_input, x2_input, x3_input, x4_input], outputs=predictions)
    
    model.compile(loss='mape', optimizer='adam', metrics=['mape']) 
    model.summary()
    return model









def get_model(inputShape):
    #return custom_cnn(inputShape)
    #return resnet_finetuning(inputShape)
    return vgg16_finetuning(inputShape)




def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




#https://stackoverflow.com/questions/49404993/keras-how-to-use-fit-generator-with-multiple-inputs/49405175
class MultipleInputGenerator(Sequence):
    """Wrapper of 4 ImageDataGenerator"""

    def __init__(self, X1, X2, X3, X4, Y, batch_size):
        # Keras generator
        self.generator = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range = 0.5
        )

        # Real time multiple input data augmentation
        self.genX1 = self.generator.flow(X1, Y, batch_size=batch_size, seed=1, shuffle=True)
        self.genX2 = self.generator.flow(X2, Y, batch_size=batch_size, seed=1, shuffle=True)
        self.genX3 = self.generator.flow(X3, Y, batch_size=batch_size, seed=1, shuffle=True)
        self.genX4 = self.generator.flow(X4, Y, batch_size=batch_size, seed=1, shuffle=True)

    def __len__(self):
        """It is mandatory to implement it on Keras Sequence"""
        return self.genX1.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        X1_batch, Y_batch = self.genX1.__getitem__(index)
        X2_batch, Y_batch = self.genX2.__getitem__(index)
        X3_batch, Y_batch = self.genX3.__getitem__(index)
        X4_batch, Y_batch = self.genX4.__getitem__(index)
        
        X_batch = [X1_batch, X2_batch, X3_batch, X4_batch]

        return X_batch, Y_batch





test_preds = pd.read_csv(inputPath+"sample_submission.csv")
test_preds['price'] = 0

mape_results = []
seed_list = [2020, 2021, 2022, 2023]
for seed in seed_list:
    seed_everything(seed)
    folds = KFold(n_splits=5, random_state=seed, shuffle=True)
    for n_, (train_idx, val_idx) in enumerate(folds.split(train)):
        print("folds: ", n_)
        train_x = train.iloc[train_idx,:]
        valid_x = train.iloc[val_idx,:]
        train_images1_x = train_images1[train_idx,:]
        valid_images1_x = train_images1[val_idx,:]
        train_images2_x = train_images2[train_idx,:]
        valid_images2_x = train_images2[val_idx,:]
        train_images3_x = train_images3[train_idx,:]
        valid_images3_x = train_images3[val_idx,:]
        train_images4_x = train_images4[train_idx,:]
        valid_images4_x = train_images4[val_idx,:]

        train_y = train_x['price'].values
        valid_y = valid_x['price'].values

        train_x = train_x.drop(['price'], axis=1)
        valid_x = valid_x.drop(['price'], axis=1)

        # callback parameter
        filepath = f'transfer_learning_best_model_{n_}.hdf5'
        es = EarlyStopping(patience=10, mode='min', verbose=1) 
        checkpoint = ModelCheckpoint(monitor='val_loss', filepath=filepath, save_best_only=True, mode='auto') 
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',  patience=5, verbose=1,  mode='min')

        """
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range = 0.5
        )
        """
        # 訓練実行
        inputShape = (size, size, 3)
        model = get_model(inputShape)
        # train_datagen = datagen.flow(train_images_x, train_y, batch_size=batch_size, shuffle=True)
        train_datagen = MultipleInputGenerator(
            X1=train_images1_x,
            X2=train_images2_x,
            X3=train_images3_x,
            X4=train_images4_x, 
            Y=train_y,
            batch_size=batch_size
        )
        history = model.fit_generator(
            train_datagen,
            validation_data=([valid_images1_x, valid_images2_x, valid_images3_x, valid_images4_x], valid_y),
            steps_per_epoch=len(train_images1_x) / batch_size,
            epochs=100,
            callbacks=[es, checkpoint, reduce_lr_loss]
        )

        # load best model weights
        model.load_weights(filepath)

        # 評価
        valid_pred = model.predict([valid_images1_x, valid_images2_x, valid_images3_x, valid_images4_x], batch_size=32).reshape((-1,1))
        test_preds['price'] += model.predict([test_images1, test_images2, test_images3, test_images4], batch_size=32).reshape((-1)) / (5*len(seed_list))
        mape_score = mean_absolute_percentage_error(valid_y, valid_pred)
        mape_results.append(mape_score)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.plot(epochs, loss, 'bo' ,label = 'training loss')
        plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.show()
        print (mape_score)
    print('mape_mean: ', np.mean(mape_results))




print('mape_mean: ', np.mean(mape_results))




test_preds




test_preds.to_csv('submit.csv', index=False)




# TODO
## 画像の結合
## data augmentation
## テーブルの利用

