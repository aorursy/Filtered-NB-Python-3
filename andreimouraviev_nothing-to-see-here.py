#!/usr/bin/env python
# coding: utf-8



import numpy as np
import os
import keras
import pandas as pd
import matplotlib.pyplot as plt




#Import Keras
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')




# let's get the mnist data
test_set=pd.read_csv(r'test.csv')
train_set=pd.read_csv('train.csv')
test_set=test_set.as_matrix()
train_set=train_set.as_matrix()



print('train set {0}, test set {1}'.format(train_set.shape,test_set.shape))




#split into labels and images
labels = train_set[:,0]
imgs = train_set[:,1:]

#split training data for training and validation
ntrain = 40000
train_labels,val_labels=labels[0:ntrain],labels[ntrain:]
train_imgs,val_imgs=imgs[0:ntrain,:],imgs[ntrain:,:]

n=1
print('true label: {}'.format(labels[n]) )
plt.imshow(imgs[n,:].reshape((28,28) ) )
plt.show()


print(  'img data shape: train set {0} validation set {1}'.format(     train_imgs.shape,val_imgs.shape)) 
print(  'label data shape: train set {0} validation set {1}'.format(     train_labels.shape,val_labels.shape))
print('data type: {0}, max val: {1}, min val: {2}'.format(train_imgs.dtype,train_imgs.max(),train_imgs.min()) )
print('data mean {0}, data std {1}'.format(np.mean(train_imgs) , np.std(train_imgs) ) )




# reshape data for CNN
train_imgs_k,val_imgs_k = np.reshape(train_imgs,(train_imgs.shape[0],28,28,1)),np.reshape(val_imgs,(val_imgs.shape[0],28,28,1))
test_imgs_k = np.reshape(test_set,(test_set.shape[0],28,28,1))
train_labels_k,val_labels_k=    keras.utils.to_categorical(train_labels, num_classes=10),keras.utils.to_categorical(val_labels, num_classes=10)

print('imgs reshaped for CNN train/val: {0} / {1}'.format(train_imgs_k.shape,val_imgs_k.shape) )
print('one-hot labels for CNN train/val: {0} / {1}'.format(train_labels_k.shape,val_labels_k.shape) )




#Define some usefull functions
def get_CNN(img_rows, img_cols,LossF ,Metrics,Optimizer=Adam(1e-5),DropP1=0.,DropP2=0.,reg=0.000,batch_norm=False,Activation='relu'):
    print('Optimizer: {0}, DropPerct: {1}, Loss: {2}, reg: {3} batch norm: {4}'.format(Optimizer, (DropP1,DropP2), LossF, reg, batch_norm))
    L2 = l2(reg)
    
    
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation=Activation,kernel_regularizer=L2, input_shape=(img_rows, img_cols, 1)))
    model.add(Conv2D(32, (3, 3), activation=Activation,kernel_regularizer=L2))
    model.add(Conv2D(32, (3, 3), activation=Activation,kernel_regularizer=L2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if batch_norm: model.add(BatchNormalization())
    model.add(Dropout(DropP1))

    model.add(Conv2D(64, (3, 3), activation=Activation,kernel_regularizer=L2))
    model.add(Conv2D(64, (3, 3), activation=Activation,kernel_regularizer=L2))
    model.add(Conv2D(32, (3, 3), activation=Activation,kernel_regularizer=L2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if batch_norm: model.add(BatchNormalization())
    model.add(Dropout(DropP1))

    model.add(Flatten())
    model.add(Dense(256, activation=Activation,kernel_regularizer=L2))
    model.add(Dense(256, activation=Activation,kernel_regularizer=L2))
    if batch_norm: model.add(BatchNormalization())
    model.add(Dropout(DropP2))
    model.add(Dense(10, activation='softmax',kernel_regularizer=L2))
    
    model.compile(loss='categorical_crossentropy', optimizer=Optimizer)
    return model

def evaluate(X,y,model,S=0):
    print '\n validation scores: \n'
    scores = model.evaluate(X,y, batch_size=1, verbose=0)
    if S: txtfile = open('model_eval{0}.txt'.format(i),'w')
    if S: txtfile.write('MODEL EVAL - Patients {0} \r\n'.format(patients))
    for score,metric in zip(scores,model.metrics_names):
        print '{0} score: {1}'.format(metric,score)
        if S: txtfile.write('{0} score: {1} \r\n'.format(metric,score))
    if S: txtfile.close() 
    return scores,model.metrics_names

def plot_loss(hist):
    val_loss = hist['val_loss']
    loss = hist['loss']

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(loss)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss ')
    ax1.set_title('training')
    
    ax2.plot(val_loss)
    ax2.set_title('validation')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss ')
    
    plt.show()
    return

def plot_loss_tail(hist):
    val_loss = hist['val_loss'][-20:-1]
    loss = hist['loss'][-20:-1]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(loss)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss ')
    ax1.set_title('training')
    
    ax2.plot(val_loss)
    ax2.set_title('validation')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss ')
    
    plt.show()
    return









#Set model Parameters
img_rows,img_cols = 28,28 # image resolution
LossF=categorical_crossentropy
Metrics=[categorical_accuracy]
Activation = 'relu'
batchNorm= True
Lr = 1.0e-1
Optim= SGD(lr=Lr, decay=1e-6, momentum=0.9, nesterov=True) # Adam(lr) 
cnn_model = get_CNN(img_rows,img_cols,LossF ,Metrics,Optimizer=Optim,DropP1=0.04,DropP2=0.4,reg=0.0005,batch_norm=batchNorm,Activation=Activation)

#Set training parameters
N_epochs = 250
Batch_Size = 1500 # adjust to fit gpu limitations
reduce_lr1=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.01, cooldown=1000, min_lr=0)
reduce_lr2=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.005, cooldown=1000, min_lr=0)
reduce_lr3=keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, verbose=1, mode='auto', epsilon=0.001, cooldown=1000, min_lr=0)


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0., patience=15, verbose=1, mode='auto')
model_checkpoint = ModelCheckpoint('cnn.hdf5', monitor='val_loss', save_best_only=True)
CALLBACKS = [early_stop,model_checkpoint,reduce_lr1,reduce_lr2,reduce_lr3]





#set parameters for on-line data augmentation
train_datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=5.,
    width_shift_range=.05,
    height_shift_range=.05,
    shear_range=20*(3.1416/180),
    zoom_range=0.05,
    data_format="channels_last")


train_datagen.fit(train_imgs_k)




'''
history = cnn_model.fit(train_imgs_k, train_labels_k, batch_size=Batch_Size, epochs=N_epochs,\
                 verbose=True, shuffle=True,callbacks=CALLBACKS,\
                 validation_data=(val_imgs_k,val_labels_k))
'''




# Train model with data (using on line augmentations)
Steps = len(train_imgs_k)/Batch_Size
hist=cnn_model.fit_generator(train_datagen.flow(train_imgs_k,train_labels_k, batch_size=Batch_Size,shuffle=True), steps_per_epoch=Steps, epochs=N_epochs,                 verbose=True,callbacks=CALLBACKS,                 validation_data=(val_imgs_k,val_labels_k))




#Plot Training Loss
plot_loss(hist.history )
plot_loss_tail(hist.history )




# Sample a few predictions for sanity check
def sample_results(imgs,labels,model,n=10):
    N = imgs.shape[0]
    samples = np.random.randint(0,high=N,size=n)
    y_pred = model.predict(imgs)
    for s in samples:
        img,lbl,pred=imgs[s],labels[s],y_pred[s]
        print('y_true: {0}, y_pred: {1}'.format(np.argmax(lbl),np.argmax(pred) ) )
        
sample_results(val_imgs_k,val_labels_k,cnn_model)




# Generate prediction accuracy for validation data
pred_proba = cnn_model.predict(val_imgs_k)
y_pred=np.argmax(pred_proba,axis=1)
y_true = np.argmax(val_labels_k,axis=1)
cnn_accuracy = accuracy_score(y_true, y_pred, normalize=True)
print(cnn_accuracy)




#predict labels for test images
pred_proba_test = cnn_model.predict(test_imgs_k)
y_pred_test=np.argmax(pred_proba_test,axis=1)




# Write to file
fid = open('mnist_sub.csv','w')
fid.write('ImageId,Label\r\n')
for i,l in enumerate(y_pred_test):
    fid.write('{0},{1}\r\n'.format(i+1,l))
fid.close()

