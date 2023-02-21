#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(2)

import matplotlib.pyplot as plt

import seaborn as seaborn
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load the data
train = pd.read_csv("../input/sample.csv",index_col=False,header=None)
# drop 2 extreme outliners
train.drop(labels=[29115,52648],axis=0,inplace=True)
train.info()





train2=train.rename(columns={295:'labels'})
train2.head()
train2['labels'] = pd.Categorical(train2['labels'])
train2[296] = train2['labels'].cat.codes
train2.head()
#correlation map
f,ax = plt.subplots(figsize=(10, 10))
seaborn.heatmap(train2[[296,3, 43, 64, 294,4,23,36]].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)




# labels are highly imbalanced, 70% is 'C'. 
Y_train = train[295]

# Drop 'label' column
X_train = train.drop(labels = [295],axis = 1) 

g = seaborn.countplot(Y_train)
Y_train.value_counts()




#there are 4 float64 columns, all are possitive
float_cols = [col for col in train.columns
              if(train[col].dtype == np.float64)]

#Standardization refers to shifting the distribution of each attribute to have 
#a mean of zero and a standard deviation of one (unit variance). 
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X_float);




from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')




print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))




# 288 colums are binary
bool_cols = [col for col in X_train
             if X_train[[col]].isin([0, 1]).all().values]
len(bool_cols)




# there are 3 int columns that I consider multi-catergorical 
multiclass_cols=X_train.columns[X_train.columns.isin(bool_cols+float_cols) == False].values
#some of them have negative values
(X_train[multiclass_cols]<0).sum()




# value distribution
X_train[4].value_counts()




X_train[23].value_counts()





X_train[36].value_counts()




# I feature engineer them to catergorical cloumns
X_train['col4a'] = X_train[4].map(lambda s: 1 if s == 0 else 0)
X_train['col4b'] = X_train[4].map(lambda s: 1 if 1<=s<=2 else 0)
X_train['col4c'] = X_train[4].map(lambda s: 1 if 3<=s<=4 else 0)
X_train['col4d'] = X_train[4].map(lambda s: 1 if 5<=s else 0)
X_train.head()




# I feature engineer them to catergorical cloumns
X_train['col23a'] = X_train[23].map(lambda s: 1 if s == -1 else 0)
X_train['col23b'] = X_train[23].map(lambda s: 1 if 1<=s<=4 else 0)
X_train['col23c'] = X_train[23].map(lambda s: 1 if 5<=s else 0)

# I feature engineer them to catergorical cloumns
X_train['col36a'] = X_train[36].map(lambda s: 1 if s == 1 else 0)
X_train['col36b'] = X_train[36].map(lambda s: 1 if (s == 0 or s==3) else 0)
X_train['col36c'] = X_train[36].map(lambda s: 1 if s==2 else 0)
X_train['col36d'] = X_train[36].map(lambda s: 1 if (4<=s<=7) else 0)
X_train['col36e'] = X_train[36].map(lambda s: 1 if (8<=s or s<0) else 0)

X_train.drop(labels=[4,23,36], axis =1, inplace=True)




X_train.shape




X_train.to_csv('X_train.csv',index=False)
Y_train.to_csv('Y_train.csv',index=False)




X_sample=X_train.ix[np.random.choice(X_train.index, 10)].transpose()
X_sample.shape




X_sample.head()




X_sample.hist(layout=(2,5))  




Zero_test=(X_train==0)
Zero_test.describe()




(X_train > 0).shape




d=pd.DataFrame({'a': (1,2,5,3),
               'b': (4,7,1,0)})
d.head()
c= ((d>0).sum(0)) 
c.head()
c[c>3]




d.drop(labels=[1],axis=0,inplace=True)
d.head()





fig, ax = plt.subplots(1,1)
ax.plot((X_train>0).sum(0), color='g', label="Number of non-zero rows per column")

legend = ax.legend()




((X_train>0).sum(0) > 0).sum()














Non_zero_count = ((X_train>0).sum(0)) 

Zero_filter=Non_zero_count[Non_zero_count > 10000 ].index.values
Zero_filter




len(Zero_filter)




#X_train.drop(labels=Zero_filter_list,axis=1,inplace=True)
X_train_zero_filter=X_train[Zero_filter]
X_train_zero_filter.shape




MeanRow=X_train_zero_filter.mean(axis=1)
MedianRow=X_train_zero_filter.median(axis=1)
MaxRow=X_train_zero_filter.max(axis=1)
MinRow=X_train_zero_filter.min(axis=1)
fig, ax = plt.subplots(1,1)
ax.plot(MeanRow, color='b', label="Mean")
ax.plot(MedianRow, color='r', label="Median")
legend = ax.legend()




X_sample=X_train_zero_filter.ix[np.random.choice(X_train.index, 10)].transpose()
X_sample.hist(layout=(2,5))  




# Load the data
train = pd.read_csv("../input/sample.csv",index_col=False,header=None)
train.shape




train.drop(labels=[29115,52648],axis=0,inplace=True)
train.shape




train_zero_filter=train[np.append(Zero_filter, [295])]
train_zero_filter.shape




train_groupby_label=train_zero_filter.groupby(295 )
train_groupby_label.median().transpose().hist()




train_groupby_label.mean().transpose().hist()




trainA=train_zero_filter[train_zero_filter[295]=='A']
trainA_sample=trainA.loc[np.random.choice(trainA.index, 10)]
trainA_sample.drop(labels=[295],axis=1).transpose().hist(layout=(2,5))




trainC=train_zero_filter[train_zero_filter[295]=='C']
trainC_sample=trainA.loc[np.random.choice(trainA.index, 10)]
trainC_sample.drop(labels=[295],axis=1).transpose().hist(layout=(2,5))





# remove warnings
import warnings
warnings.filterwarnings('ignore')
# ---




# Load the data
train = pd.read_csv("../input/sample.csv",index_col=False,header=None)
train.drop(labels=[29115,52648],axis=0,inplace=True)
train.shape




train.head()




train.info()




float_cols = [col for col in train.columns
              if(train[col].dtype == np.float64)]
(train[float_cols]<0).sum()




Y_train = train[295]
# Drop 'label' column
X_train = train.drop(labels = [295],axis = 1) 

bool_cols = [col for col in X_train
             if X_train[[col]].isin([0, 1]).all().values]
bool_cols




multiclass_cols=X_train.columns[X_train.columns.isin(bool_cols+float_cols) == False].values
(train[multiclass_cols]<0).sum()




train[4].value_counts()




train[23].value_counts()




train[36].value_counts()




train[col] = train[col].where(train[col]==0, 1)




train[4].hist()




train[23].hist()




train[36].hist()




train[3].hist()




train[43].hist()




train[64].hist()




train[294].hist()




Y_train = train[295]

# Drop 'label' column
X_train = train.drop(labels = [295],axis = 1) 




#correlation map
f,ax = plt.subplots(figsize=(10, 10))
seaborn.heatmap(X_train[[3, 43, 64, 294]].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)




#correlation map
f,ax = plt.subplots(figsize=(10, 10))
seaborn.heatmap(X_train[[4, 23, 36]].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)




df=pd.DataFrame(np.random.randn(5, 5))
df.head(n=5)




df.loc[:,['a','c']] = df.loc[:,['a','c']]/df[['a','c']].max()
df.head()




for col in [2,4]:
    df[col] = df[col].where(df[col]==0, 1)
df.head()




train.loc[:,float_cols]=train.loc[:,float_cols]/train[float_cols].max()




train[3].hist()




train[43].hist()




train[64].hist()




train[294].hist()




for col in multiclass_cols:
    train[col] = train[col].where(train[col]==0, 1)




np.unique(train[multiclass_cols])




multiclass_cols




train[4].hist()




train[23].hist()




train[36].hist()




#correlation map
f,ax = plt.subplots(figsize=(10, 10))
seaborn.heatmap(X_train[[3, 43, 64, 294,4,23,36]].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

