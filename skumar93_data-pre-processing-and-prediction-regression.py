#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model

import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer




pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')




print("train: "+ str(train.shape))
print("test: "+ str(test.shape))




train.SalePrice.describe()




sns.distplot(train['SalePrice'])




sns.boxplot(train['SalePrice'])




# Analyze missing data
total=train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis=1,keys=['Total', 'Percent'])
print(missing_data)




#Remove features with more than 80% of missing data
toDrop = missing_data.loc[missing_data['Percent']>0.8].index
trainFinal = train.drop(toDrop,axis=1)
test = test.drop(toDrop,axis=1)
print('Dropped features:')
print(toDrop)
print(trainFinal.shape,test.shape)




# Split categorical and numerical data
categorical = [col for col in trainFinal.columns if trainFinal.dtypes[col]=='object']
quantitative = [col for col in trainFinal.columns if trainFinal.dtypes[col]=='int64']




# Create new features for categorical data
for c in categorical:
    trainFinal[c]=trainFinal[c].astype('category')
    test[c]=test[c].astype('category')
print(trainFinal.shape,test.shape)
trainFinal = pd.get_dummies(trainFinal)
test = pd.get_dummies(test)
colTrain = set(trainFinal.columns)
colTest  = set(test.columns)
colTodel = colTrain - colTest
for c in colTodel:
    test[c]= 0
print(trainFinal.shape,test.shape)




#Fill the missing vaues with mean value of the features
trainFinal = trainFinal.fillna(trainFinal.mean())
test = test.fillna(test.mean())




#Get the correlation matrix
corr= trainFinal[quantitative].corr()
corr1 = trainFinal.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr,vmax=0.8,square=True)
corrSalesprice = corr1.SalePrice
print(corrSalesprice.sort_values(ascending=False))




#get all highly inter-correlated variable and remove the redundant variables
out =pd.DataFrame(columns =('var1','var2'))
for index,rows in corr1.iterrows():
    for c in corr1.columns:
        if corr1[index][c] > 0.8 and (str(c) != str(index)):
            if corrSalesprice[c] < corrSalesprice[index]:
                tmp = c
                c= index
                index = tmp
            out =  out.append({'var1':c,'var2':index},ignore_index=True)
out = out.drop_duplicates()
print(trainFinal.shape,test.shape)
trainFinal = trainFinal.drop(out['var2'],axis=1)
test = test.drop(out['var2'],axis=1)
print(trainFinal.shape,test.shape)
print("Dropping variables:")
print(out['var2'])




# Check the heat map after removing correlated variables
quantitative = [col for col in trainFinal.columns if trainFinal.dtypes[col]=='int64']
corr= trainFinal[quantitative].corr()
sns.heatmap(corr)




#Get the pairplot of important features 
ImpFeatures = corr.SalePrice[corr.SalePrice.values>0.5].index
sns.set()
sns.pairplot(trainFinal[ImpFeatures])
plt.show();
# Get the 'GrLivArea' acatter plot
# data=pd.concat([trainFinal['SalePrice'], trainFinal['GrLivArea']],axis = 1)
# data.plot.scatter(x='GrLivArea',y ='SalePrice')




# Log transform the skewed variables
from scipy.stats import skew
skewness = trainFinal[quantitative].apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(skewness.index)
skewness.plot.bar()
# print(skewness)




trainFinal[skewness.index]= np.log1p(trainFinal[skewness.index])
skewness=skewness.drop('SalePrice')
test[skewness.index]= np.log1p(test[skewness.index])
skewness = trainFinal[skewness.index].apply(lambda x: skew(x))
skewness.plot.bar()




# From the pairplot analyze and remove the outliers-Living area
outLivArea = trainFinal[trainFinal['GrLivArea']>8.2].index
trainOut = trainFinal.loc[outLivArea]
outLivArea = trainOut[trainOut['SalePrice']<12.5].index
trainFinal = trainFinal.drop(outLivArea,axis=0)
# Analyze scatter plot after dropping outliers
data=pd.concat([trainFinal['SalePrice'], trainFinal['GrLivArea']],axis = 1)
data.plot.scatter(x='GrLivArea',y ='SalePrice')




#Outlers- TotalBsmtSF
data=pd.concat([trainFinal['SalePrice'], trainFinal['TotalBsmtSF']],axis = 1)
data.plot.scatter(x='TotalBsmtSF',y ='SalePrice')




outBsmt = trainFinal[trainFinal['TotalBsmtSF']<6].index
trainFinal = trainFinal.drop(outBsmt,axis=0)
data=pd.concat([trainFinal['SalePrice'], trainFinal['TotalBsmtSF']],axis = 1)
data.plot.scatter(x='TotalBsmtSF',y ='SalePrice')




data=pd.concat([trainFinal['SalePrice'], trainFinal['YearRemodAdd']],axis = 1)
data.plot.scatter(x='YearRemodAdd',y ='SalePrice')




# #Outliers- Year Remodeled
outyear = trainFinal[trainFinal['YearRemodAdd']<7.577].index
trainFinal = trainFinal.drop(outyear,axis=0)
print(len(outyear))
data=pd.concat([trainFinal['SalePrice'], trainFinal['YearRemodAdd']],axis = 1)
data.plot.scatter(y='SalePrice',x ='YearRemodAdd')









y = trainFinal['SalePrice']
x= trainFinal.drop('SalePrice',axis=1)
x= x.drop('Id',axis=1)
test=test.drop('Id',axis=1)
test=test.drop('SalePrice',axis=1)
scaler = StandardScaler()
x=scaler.fit_transform(x)
test = scaler.fit_transform(test)




#Split train and covariance sets
from sklearn.model_selection import cross_val_score, train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.01, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))




# Define error measure for official scoring : RMSE

scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)




# Linear Regression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

#Applying Lasso
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(x, y)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 10000, cv = 10)
lasso.fit(x, y)
alpha = lasso.alpha_
print("Best alpha :", alpha)

print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())
print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())
Ypred = np.exp(lasso.predict(test))
print(Ypred)




x.shape




Ypred = np.exp(lasso.predict(test))
print(Ypred)




start = Ypred.shape[0] + 2
out = open('output.csv', "w")
out.write("Id,SalePrice\n")
rows = ['']*Ypred.shape[0] # predefine or use append


for num in range(0, test.shape[0]):
    Id = start + num  #TODO; classify here
    rows[num] = "%d,%d\n"%(Id,Ypred[num])
    print(rows[num])
out.writelines(rows)
out.close()




y.loc[0]




trainFinal.loc[0]




rmse= np.sqrt(-cross_val_score(lasso, x, y, scoring = scorer, cv = 10))




rmse.mean()




trainFinal['SalePrice'].describe()




X_train

