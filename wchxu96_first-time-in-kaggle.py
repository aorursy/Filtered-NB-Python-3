#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')




train_data.head()




#test_data.head()
test_data.shape
train_data.shape




data = pd.concat([train_data,test_data],axis=0)
label = train_data.SalePrice
del data['SalePrice']
data.shape




data.drop('Id',axis=1,)




data.head(5)




nan = data.isnull().sum()
nan = nan[nan > 0]
#print (nan.shape)
#print (nan.head())




data_drop_na = data.dropna(axis=1,thresh=2000)




data_na_dummy = pd.get_dummies(data_drop_na)




data_na_dummy.shape




imp = Imputer(strategy='median',axis=1)
data_without_na = imp.fit_transform(data_na_dummy)




data_without_na.shape




data_without_na[:5,:15]




scale = StandardScaler().fit(data_without_na)
std_data = scale.transform(data_without_na)




ax = sns.distplot(label)




sns.distplot(np.log(label))




log_label = np.log(label)




pca = PCA()
pca.fit(std_data)
evr = np.cumsum(pca.explained_variance_ratio_)




evr[evr < 0.95].shape




pca_data = PCA(n_components=178).fit_transform(std_data)




def accuracy()

