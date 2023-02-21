#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import pandas as pd

from glob import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,                          Flatten, Convolution2D,                          Reshape
from keras.optimizers import SGD, Adam
from keras import backend as K
smooth = 1.
def dice_coef(y_true, y_pred):
    #y_true_f = K.flatten(y_true)
    #y_pred_f = K.flatten(y_pred)
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    return (2. * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
from keras.utils import np_utils

from skimage.io import imread
from sklearn.model_selection import train_test_split




# set channels first notation
K.set_image_dim_ordering('th')




# helper function to load the image and downsample it by 10
jim_fix = lambda x: np.expand_dims(x[::10, ::10],0)
jimread = lambda x: jim_fix(imread(x))




BASE_IMAGE_PATH = os.path.join('..', 'input')
all_images = glob(os.path.join(BASE_IMAGE_PATH, '2d_images', '*.tif'))
all_masks = ['_masks'.join(c_file.split('_images')) for c_file in all_images]
print(len(all_masks), 'matching files found')




get_ipython().run_line_magic('matplotlib', 'inline')
test_image = jimread(all_images[0])
test_mask = jimread(all_masks[0])
fig, (ax1 ,ax2) = plt.subplots(1, 2)
ax1.imshow(test_image[0])
ax2.imshow(test_mask[0])




print('Total samples are', len(all_images))
print('Image resolution is', test_image.shape)




images = np.stack([jimread(i) for i in all_images], 0) / 1024.0
masks = np.stack([jimread(i) for i in all_masks], 0) / 255.0
X_train, X_test, y_train,  y_test = train_test_split(images, masks, test_size=0.1, random_state=1234)
print('Training input is', X_train.shape)
print('Training output is {}, min is {}, max is {}'.format(y_train.shape, y_train.min(), y_train.max()))
print('Testing set is', X_test.shape)




# Create a deep nn
simple_fully_connected_model = Sequential()
# dense layers only work with vector input
simple_fully_connected_model.add(Flatten(input_shape=images.shape[1:]))
# add the dense (fully-connected layer) mapping each pixel to a pixel
simple_fully_connected_model.add(Dense(np.prod(images.shape[1:]), activation='sigmoid'))
# reshape it so we have an image output
simple_fully_connected_model.add(Reshape(target_shape=images.shape[1:]))

simple_fully_connected_model.compile(loss='mse',
             optimizer=Adam(lr=10e-3),
             metrics=['accuracy','mse',dice_coef])
loss_history = []
print(simple_fully_connected_model.summary())




fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[0,0])
ax1.set_title('Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(y_test[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(X_test)[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('Prediction\n{}'.format(pred_img.shape))




loss_history += [simple_fully_connected_model.fit(X_train, y_train, epochs=1, batch_size=20, validation_split = 0.1, verbose = False, shuffle = True)]




fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[0,0])
ax1.set_title('Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(y_test[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(X_test)[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('Prediction\n{}'.format(pred_img.shape))




loss_history += [simple_fully_connected_model.fit(X_train, y_train, epochs=5, batch_size=20, validation_split = 0.1, verbose = False, shuffle = True)]




fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[0,0])
ax1.set_title('Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(y_test[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(X_test)[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('Prediction\n{}'.format(pred_img.shape))




# how does it do on flipped
flip_fcn = lambda x: x.swapaxes(2,3)
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(flip_fcn(X_train)[0,0])
ax1.set_title('Flipped Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(flip_fcn(y_test)[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_train[0,0].shape))
pred_img = simple_fully_connected_model.predict(flip_fcn(X_test))[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('Prediction\n{}'.format(pred_img.shape))




get_ipython().run_line_magic('matplotlib', 'inline')
epich = np.cumsum(np.concatenate([np.linspace(0.5,1,len(mh.epoch)) for mh in loss_history]))
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (8,18))
_ = ax1.plot(epich,np.concatenate([mh.history['loss'] for mh in loss_history]),'b-',
            epich,np.concatenate([mh.history['val_loss'] for mh in loss_history]),'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

_ = ax3.semilogy(epich,np.concatenate([mh.history['mean_squared_error'] for mh in loss_history]),'b-',
    epich,np.concatenate([mh.history['val_mean_squared_error'] for mh in loss_history]),'r-')
ax3.legend(['Training', 'Validation'])
ax3.set_title('MSE')

_ = ax2.plot(epich,np.concatenate([mh.history['dice_coef'] for mh in loss_history]),'b-',
    epich,np.concatenate([mh.history['val_dice_coef'] for mh in loss_history]),'r-')
ax2.legend(['Training', 'Validation'])
ax2.set_title('Dice Coefficient')




# Create a deep nn
dual_fully_connected_model = Sequential()
# dense layers only work with vector input
dual_fully_connected_model.add(Flatten(input_shape=images.shape[1:]))
# add a very small dense layer with a relu
dual_fully_connected_model.add(Dense(128, activation='relu'))
# add the dense (fully-connected layer) mapping each pixel to a pixel
dual_fully_connected_model.add(Dense(np.prod(images.shape[1:]), activation='sigmoid'))
# reshape it so we have an image output
dual_fully_connected_model.add(Reshape(target_shape=images.shape[1:]))

dual_fully_connected_model.compile(loss='mse',
             optimizer=Adam(lr=10e-3),
             metrics=['accuracy','mse',dice_coef])
loss_history = []
print(dual_fully_connected_model.summary())




fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (12,4))
ax1.imshow(X_test[0,0])
ax1.set_title('Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(y_test[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(X_test)[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('1-Lay-Prediction\n{}'.format(pred_img.shape))
pred_img2 = dual_fully_connected_model.predict(X_test)[0,0]
ax4.imshow(pred_img2, vmin=0,vmax=1, cmap='bone')
ax4.set_title('2-Lay-Prediction\n{}'.format(pred_img.shape))




loss_history += [dual_fully_connected_model.fit(X_train, y_train, epochs=10, batch_size=20, validation_split = 0.1, verbose = False, shuffle = True)]




fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (12,4))
ax1.imshow(X_test[0,0])
ax1.set_title('Validation Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(y_test[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(X_test)[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('1-Lay-Prediction\n{}'.format(pred_img.shape))
pred_img2 = dual_fully_connected_model.predict(X_test)[0,0]
ax4.imshow(pred_img2, vmin=0,vmax=1, cmap='bone')
ax4.set_title('2-Lay-Prediction\n{}'.format(pred_img.shape))




get_ipython().run_line_magic('matplotlib', 'inline')
epich = np.cumsum(np.concatenate([np.linspace(0.5,1,len(mh.epoch)) for mh in loss_history]))
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (8,18))
_ = ax1.plot(epich,np.concatenate([mh.history['loss'] for mh in loss_history]),'b-',
            epich,np.concatenate([mh.history['val_loss'] for mh in loss_history]),'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

_ = ax3.semilogy(epich,np.concatenate([mh.history['mean_squared_error'] for mh in loss_history]),'b-',
    epich,np.concatenate([mh.history['val_mean_squared_error'] for mh in loss_history]),'r-')
ax3.legend(['Training', 'Validation'])
ax3.set_title('MSE')

_ = ax2.plot(epich,np.concatenate([mh.history['dice_coef'] for mh in loss_history]),'b-',
    epich,np.concatenate([mh.history['val_dice_coef'] for mh in loss_history]),'r-')
ax2.legend(['Training', 'Validation'])
ax2.set_title('Dice Coefficient')




get_ipython().run_cell_magic('time', '', 'loss_history += [dual_fully_connected_model.fit(X_train, y_train, epochs=60, batch_size=20, validation_data = (X_test, y_test), verbose = False, shuffle = True)]')




get_ipython().run_line_magic('matplotlib', 'inline')
epich = np.cumsum(np.concatenate([np.linspace(0.5,1,len(mh.epoch)) for mh in loss_history]))
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (8,18))
_ = ax1.plot(epich,np.concatenate([mh.history['loss'] for mh in loss_history]),'b-',
            epich,np.concatenate([mh.history['val_loss'] for mh in loss_history]),'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

_ = ax3.semilogy(epich,np.concatenate([mh.history['mean_squared_error'] for mh in loss_history]),'b-',
    epich,np.concatenate([mh.history['val_mean_squared_error'] for mh in loss_history]),'r-')
ax3.legend(['Training', 'Validation'])
ax3.set_title('MSE')

_ = ax2.plot(epich,np.concatenate([mh.history['dice_coef'] for mh in loss_history]),'b-',
    epich,np.concatenate([mh.history['val_dice_coef'] for mh in loss_history]),'r-')
ax2.legend(['Training', 'Validation'])
ax2.set_title('Dice Coefficient')




fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (12,4))
ax1.imshow(X_test[0,0])
ax1.set_title('Validation Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(y_test[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(X_test)[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('1-Lay-Prediction\n{}'.format(pred_img.shape))
pred_img2 = dual_fully_connected_model.predict(X_test)[0,0]
ax4.imshow(pred_img2, vmin=0,vmax=1, cmap='bone')
ax4.set_title('2-Lay-Prediction\n{}'.format(pred_img.shape))




fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (12,4))
ax1.imshow(flip_fcn(X_test)[0,0])
ax1.set_title('Flipped Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(flip_fcn(y_test)[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(flip_fcn(X_test))[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('1-Lay-Prediction\n{}'.format(pred_img.shape))
pred_img2 = dual_fully_connected_model.predict(flip_fcn(X_test))[0,0]
ax4.imshow(pred_img2, vmin=0,vmax=1, cmap='bone')
ax4.set_title('2-Lay-Prediction\n{}'.format(pred_img.shape))




get_ipython().run_cell_magic('time', '', 'loss_history += [dual_fully_connected_model.fit(flip_fcn(X_train), flip_fcn(y_train), epochs=20, batch_size=20, validation_data = (flip_fcn(X_test), flip_fcn(y_test)), verbose = False, shuffle = True)]')




fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (12,4))
ax1.imshow(flip_fcn(X_test)[0,0])
ax1.set_title('Flipped Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(flip_fcn(y_test)[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(flip_fcn(X_test))[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('1-Lay-Prediction\n{}'.format(pred_img.shape))
pred_img2 = dual_fully_connected_model.predict(flip_fcn(X_test))[0,0]
ax4.imshow(pred_img2, vmin=0,vmax=1, cmap='bone')
ax4.set_title('2-Lay-Prediction\n{}'.format(pred_img.shape))




fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (12,4))
ax1.imshow(X_test[0,0])
ax1.set_title('Validation Input\n{}'.format(X_test[0,0].shape))
ax2.imshow(y_test[0,0],vmin=0,vmax=1,cmap='bone')
ax2.set_title('Ground Truth\n{}'.format(y_test[0,0].shape))
pred_img = simple_fully_connected_model.predict(X_test)[0,0]
ax3.imshow(pred_img, vmin=0,vmax=1, cmap='bone')
ax3.set_title('1-Lay-Prediction\n{}'.format(pred_img.shape))
pred_img2 = dual_fully_connected_model.predict(X_test)[0,0]
ax4.imshow(pred_img2, vmin=0,vmax=1, cmap='bone')
ax4.set_title('2-Lay-Prediction\n{}'.format(pred_img.shape))




import nibabel as nib
from glob import glob
all_nifti_imgs = glob(os.path.join('..', 'input', '3d_images', 'IMG_*.nii.gz'))
nifti_img = nib.load(all_nifti_imgs[0])
n_stack = np.stack([jim_fix(img) for img in nifti_img.get_data()[::2]],0)
pred_stack = simple_fully_connected_model.predict(n_stack)
pred_stack2 = dual_fully_connected_model.predict(n_stack)




from skimage.util.montage import montage2d
fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize = (9,16))
ax1.imshow(montage2d(n_stack[:,0]), cmap = 'bone')
ax1.axis('off')
ax1.set_title('Image Input')
ax2.imshow(montage2d(pred_stack[:,0]))
ax2.set_title('Prediction (1 layer)')
ax2.axis('off')
ax3.imshow(montage2d(pred_stack2[:,0]))
ax3.set_title('Prediction (2 layer)')
ax3.axis('off')




from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
p = pred_stack[::-1,0].swapaxes(1,2)
cmap = plt.cm.get_cmap('nipy_spectral_r')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

verts, faces = measure.marching_cubes(p, 0)
mesh = Poly3DCollection(verts[faces], alpha=0.25, edgecolor='none', linewidth = 0.1)

mesh.set_edgecolor([1, 0, 0])
ax.add_collection3d(mesh)

ax.set_xlim(0, p.shape[0])
ax.set_ylim(0, p.shape[1])
ax.set_zlim(0, p.shape[2])

ax.view_init(45, 45)
fig.savefig('lung_3d.pdf')






