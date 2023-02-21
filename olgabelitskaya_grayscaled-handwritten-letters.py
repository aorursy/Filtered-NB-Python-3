#!/usr/bin/env python
# coding: utf-8



import warnings; warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,pylab as pl,h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from IPython import display
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from keras.models import Sequential,load_model,Model
from keras.layers import Dense,LSTM,GlobalAveragePooling1D,GlobalAveragePooling2D
from keras.layers.advanced_activations import PReLU,LeakyReLU
from keras.layers import Input,Activation,Flatten,Dropout,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D,GlobalMaxPooling2D
fw='weights.best.letters.hdf5'




# plotting of fitting histories for neural networks
def history_plot(fit_history):
    pl.figure(figsize=(12,10)); pl.subplot(211)
    keys=list(fit_history.history.keys())[0:4]
    pl.plot(fit_history.history[keys[0]],
            color='slategray',label='train')
    pl.plot(fit_history.history[keys[2]],
            color='#4876ff',label='valid')
    pl.xlabel("Epochs"); pl.ylabel("Loss")
    pl.legend(); pl.grid()
    pl.title('Loss Function')     
    pl.subplot(212)
    pl.plot(fit_history.history[keys[1]],
            color='slategray',label='train')
    pl.plot(fit_history.history[keys[3]],
            color='#4876ff',label='valid')
    pl.xlabel("Epochs"); pl.ylabel("Accuracy")    
    pl.legend(); pl.grid()
    pl.title('Accuracy'); pl.show()
# preprocessing functions 
def ohe(x): 
    return OneHotEncoder(categories='auto')           .fit(x.reshape(-1,1)).transform(x.reshape(-1,1))           .toarray().astype('int64')
def tts(X,y): 
    x_train,x_test,y_train,y_test=    train_test_split(X,y,test_size=.2,random_state=1)
    n=int(len(x_test)/2)
    x_valid,y_valid=x_test[:n],y_test[:n]
    x_test,y_test=x_test[n:],y_test[n:]
    return x_train,x_valid,x_test,y_train,y_valid,y_test




f=h5py.File('../input/LetterColorImages_123.h5','r')
keys=list(f.keys()); keys 




# creating image arrays and targets
letters=u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
backgrounds=np.array(f[keys[0]])
labels=np.array(f[keys[2]])
# normalization of image arrays
images=np.array(f[keys[1]])/255




gray_images=np.dot(images[...,:3],[.299,.587,.114])
rn=np.random.randint(10000); pl.figure(figsize=(2,3))
pl.title('Label: %s \n'%letters[labels[rn]-1]+         'Background: %s'%backgrounds[rn],
         fontsize=18)
pl.imshow(gray_images[rn],cmap=pl.cm.bone)
pl.xticks([]); pl.yticks([]); pl.show()
gray_images=gray_images.reshape(-1,32,32,1)




# one-hot encoding
cbackgrounds,clabels=ohe(backgrounds),ohe(labels)
ctargets=np.concatenate((clabels,cbackgrounds),axis=1)
display.display(pd.DataFrame([labels[97:103],clabels[97:103]]).T)
pd.DataFrame([clabels.shape,cbackgrounds.shape,ctargets.shape])




# splitting the data
x_train1,x_valid1,x_test1,y_train1,y_valid1,y_test1=tts(gray_images,clabels)
x_train2,x_valid2,x_test2,y_train2,y_valid2,y_test2=tts(gray_images,ctargets)
y_train2_list=[y_train2[:,:33],y_train2[:,33:]]
y_test2_list=[y_test2[:,:33],y_test2[:,33:]]
y_valid2_list=[y_valid2[:,:33],y_valid2[:,33:]]




def top_3_categorical_accuracy(y_true,y_pred):
    return top_k_categorical_accuracy(y_true,y_pred,k=3)
def gray_model():
    model = Sequential()    
    model.add(Conv2D(32,(5,5),padding='same',
                     input_shape=x_train1.shape[1:]))
    model.add(LeakyReLU(alpha=.02))    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.25))
    model.add(Conv2D(128,(5,5)))
    model.add(LeakyReLU(alpha=.02))    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.25))   
    model.add(GlobalMaxPooling2D())  
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=.02)) 
    model.add(Dropout(.25))  
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=.02)) 
    model.add(Dropout(.25))    
    model.add(Dense(33))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',
                  metrics=[categorical_accuracy,top_3_categorical_accuracy])
    return model
gray_model=gray_model()




checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss', 
                               patience=10,verbose=2,factor=.5)
estopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
history=gray_model.fit(x_train1,y_train1, 
                       epochs=200,batch_size=64,verbose=2,
                       validation_data=(x_valid1,y_valid1),
                       callbacks=[checkpointer,lr_reduction,estopping])




history_plot(history)
# loading the model weights with the best validation accuracy
gray_model.load_weights(fw)
# calculation classification accuracy on the testing set
gray_model.evaluate(x_test1,y_test1)




steps,epochs=1000,10
igen=ImageDataGenerator(zoom_range=.3,shear_range=.3,rotation_range=30)
generator=gray_model.fit_generator(igen.flow(x_train1,y_train1,batch_size=64),
                         steps_per_epoch=steps,epochs=epochs,verbose=2,
                         validation_data=(x_valid1,y_valid1), 
                         callbacks=[checkpointer,lr_reduction])




history_plot(generator)
gray_model.load_weights(fw)
gray_model.evaluate(x_test1,y_test1)




py_test1=gray_model.predict_classes(x_test1)
fig=pl.figure(figsize=(12,12))
for i,idx in enumerate(np.random.choice(x_test1.shape[0],
                                        size=16,replace=False)):
    ax=fig.add_subplot(4,4,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_test1[idx]),cmap=pl.cm.bone)
    pred_idx=py_test1[idx]
    true_idx=np.argmax(y_test1[idx])
    ax.set_title("{} ({})".format(letters[pred_idx],letters[true_idx]),
                 color=("darkblue" if pred_idx==true_idx else "darkred"))




def gray_multi_model():    
    model_input=Input(shape=(32,32,1))
    x=BatchNormalization()(model_input)
    x=Conv2D(32,(5,5),padding='same')(model_input)
    x=LeakyReLU(alpha=.02)(x)
    x=MaxPooling2D(pool_size=(2,2))(x)    
    x=Dropout(.25)(x)   
    x=Conv2D(256,(5,5),padding='same')(x) 
    x=LeakyReLU(alpha=.02)(x)
    x=MaxPooling2D(pool_size=(2,2))(x)    
    x=Dropout(.25)(x)             
    x=GlobalMaxPooling2D()(x)    
    x=Dense(1024)(x) 
    x=LeakyReLU(alpha=.02)(x)
    x=Dropout(.25)(x)   
    x=Dense(256)(x) 
    x=LeakyReLU(alpha=.02)(x)
    x=Dropout(.025)(x)   
    y1=Dense(33,activation='softmax')(x)
    y2=Dense(4,activation='softmax')(x)      
    model=Model(inputs=model_input,outputs=[y1,y2])
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',
                  metrics=[categorical_accuracy,top_3_categorical_accuracy])      
    return model
gray_multi_model=gray_multi_model()




checkpointer=ModelCheckpoint(filepath=fw,verbose=2,save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=2,factor=.75)
estopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
history=gray_multi_model.fit(x_train2,y_train2_list,
                             epochs=200,batch_size=128,verbose=2,
                             validation_data=(x_valid2,y_valid2_list),
                             callbacks=[checkpointer,lr_reduction,estopping])




gray_multi_model.evaluate(x_test2,y_test2_list)

