#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
import scipy
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')




labeled_images = pd.read_csv('../input/train.csv')
clf = svm.SVC()




images = labeled_images.iloc[0:5000,1:].copy()
labels = labeled_images.iloc[0:5000,:1].copy()
#make black-and-white images
images[images>0] = 1

train_images, test_images,train_labels, test_labels =     train_test_split(images, labels, train_size=0.8, random_state=0)
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images,test_labels))




score = []
clf_time = []
nips = 400
nimages = range(nips,10001,nips)
labels = labeled_images.iloc[0:10000,:1].copy()
images = labeled_images.iloc[0:10000,1:].copy()
images[images>0] = 1

for i in nimages:
    start = time.time()
    train_images, test_images,train_labels, test_labels =         train_test_split(images[0:i], labels[0:i], train_size=0.8, random_state=0)
    clf.fit(train_images, train_labels.values.ravel())
    clf_time.append(time.time()-start)
    s = clf.score(test_images,test_labels)
    score.append(s)
    print(i,s)




t = np.array(clf_time)
s = np.array(score)
n = np.array(nimages)
plt.plot(n, s)
plt.ylabel('Score')
plt.xlabel('Training Images Used')
plt.show()




def crop_img(in_img):
    sq_img = np.array(in_img.values.reshape((28,28)))
    
    x = sq_img.sum(axis=1)
    xn = np.array(np.where(x>0))
    xm = int((xn.max()-xn.min())/2)
    xc = int((xn.max()+xn.min())/2)+1

    y = sq_img.sum(axis=0)
    yn = np.array(np.where(y>0))
    ym = int((yn.max()-yn.min())/2)
    yc = int((yn.max()+yn.min())/2)+1

    hnis = np.max([xm,ym])+1
    cimg = scipy.misc.imresize(sq_img[xc-hnis:xc+hnis,yc-hnis:yc+hnis],[28,28])
    cimg[cimg>0]=1
    return cimg.reshape(784)

#th = threshold level for converting to black-and-white
def fast_crop(in_img,th):
    nipc = 25 #nipc = number of images per crop
    nc = int(in_img.shape[0]/nipc)
    out_img = in_img.copy()
    for i in range(nc):
        im = in_img.iloc[i*nipc:(i+1)*nipc,:].copy()
        im[im/256 < th] = 0
        im[im > 0] = 1
        for j in range(nipc):
            im.iloc[j] = crop_img(im.iloc[j])
        out_img.iloc[i*nipc:(i+1)*nipc,:] = im
    return out_img




images = labeled_images.iloc[0:5000,1:].copy()
labels = labeled_images.iloc[0:5000,:1].copy()
images = fast_crop(images,0)
plt.imshow(np.array(images.iloc[0]).reshape((28,28)))
plt.title('Centered and Cropped Image')

train_images, test_images,train_labels, test_labels =     train_test_split(images, labels, train_size=0.8, random_state=0)
clf.fit(train_images, train_labels.values.ravel())




print(clf.score(test_images,test_labels))




im = labeled_images.iloc[6889,1:].copy()
plt.imshow(np.array(im).reshape((28,28)))
plt.title('Original Image[6889]')
plt.show()

import copy
im2 = im.copy()
im2[im2 > 0] = 1
plt.title('Black-and-White Image[6889]')
plt.imshow(np.array(im2).reshape((28,28)))
plt.show()

im3 = im.copy()
im3[im3/256. < 0.15] = 0
im3[im3 > 0] = 1
plt.title('Black-and-White Image[6889] \nwith 15% Threshold')
plt.imshow(np.array(im3).reshape((28,28)))
plt.show()




images = labeled_images.iloc[0:5000,1:].copy()
images = fast_crop(images,0.15)
plt.imshow(np.array(images.iloc[4000]).reshape((28,28)))
plt.title('Centered and Cropped Image')
plt.show()

train_images, test_images,train_labels, test_labels =     train_test_split(images, labels, train_size=0.8, random_state=0)
clf.fit(train_images, train_labels.values.ravel())




print(clf.score(test_images,test_labels))




th = np.arange(0.2,1,0.1)
images = labeled_images.iloc[0:3000,1:].copy()
labels = labeled_images.iloc[0:3000,:1].copy()

for i in range(8):
    cc_im = fast_crop(images,th[i])
    train_images, test_images,train_labels, test_labels =         train_test_split(cc_im, labels, train_size=0.8, random_state=0)
    clf.fit(train_images, train_labels.values.ravel())
    plt.imshow(np.array(cc_im.iloc[106]).reshape((28,28)))
    title_str = 'Threshold: '+str(th[i])+'\nAccuracy: '+str(clf.score(test_images,test_labels))
    plt.title(title_str)
    plt.show()

    




#cropped/centered images
cc_images = fast_crop(labeled_images.iloc[:,1:].copy(),0.5)

#save images
cc_images.to_csv('cropped.csv', index=False)




labels = labeled_images.iloc[:,:1].copy()

train_images, test_images,train_labels, test_labels =     train_test_split(cc_images, labels, train_size=0.8, random_state=0)
clf.fit(train_images, train_labels.values.ravel())




print(clf.score(test_images,test_labels))




test_data = pd.read_csv('../input/test.csv')
cc_testdata = fast_crop(test_data.iloc[0:5000].copy(),0.5)

results=clf.predict(cc_testdata)




results




df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)

