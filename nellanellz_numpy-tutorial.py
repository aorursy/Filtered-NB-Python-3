#!/usr/bin/env python
# coding: utf-8



ds.itemsize




a = np.arange(5)
b = np.array([2,4,0,1,2])
a




diff = a-b
diff




b**2




2*b




np.sin(a)




np.sum(a)




np.max(a)




np.min(a)




b > 2




a*b




x = np.array([[1,1],[0,1]])
y = np.array([[2,0], [3,4]])
x*y




x.dot(y)




x.sum()




x.sum(axis=0)




x.sum(axis=1)




z = np.random.random((3,4))
z




np.mean(z)




np.median(z)




np.std(z)




data_set = np.random.random((2,3))
data_set




np.reshape(data_set, (3,2))




np.reshape(data_set, (6,1))




np.ravel(data_set)




np.reshape(data_set, (6))




data_set = np.random.random((5,10))
data_set




data_set[1]




data_set[1][0]




data_set[1,0]




data_set[2:4]




data_set[2:4,0]




data_set[2:4][0:2]




data_set[2:4][0]




data_set[:,0]




data_set[2:4:1]




data_set[::]




data_set[::2]




data_set[2:5:2]




data_set[2:4]
data_set[2:4, ::2]




ds.dtype




ds.shape




ds.size




threeD = np.arange(1,30,2).reshape(3,5)
print(threeD)




ds = np.arange(1,10,3)
print(ds)
ds.ndim




np.random.random((3,3))




np.linspace(2,4,10)




np.arange(1,10)




np.array([(1,3,5), (7,9,11), (13,15,17)]).




np.empty((2,2,4))




np.ones((3,3), dtype=np.int16)




np.zeros((2,3), dtype=np.int16)




type(x)




x = np.array([1,3,5,7])
print(x)




np.arange(1,20,3,dtype=np.float64)




np.arange(1,10,2)




np.arange(10)




import numpy as np






