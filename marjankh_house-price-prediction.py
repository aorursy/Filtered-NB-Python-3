#!/usr/bin/env python
# coding: utf-8



# Regression problem : target variable is continuous




import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#import os
#print(os.listdir("../input")) 




train.head()
#train.columns
#train.info()
#train.describe()




# Distribution of the target variable: SalesPrice
sns.distplot(train.SalePrice, bins = 25)




# Logarithimic distribution of the target variable: to make this distribution more symmetrical
sns.distplot(np.log(train.SalePrice), bins = 25)




test.head() 
#train.describe()




plt.xlabel("Fireplaces")
plt.ylabel("SalePrice($)")
plt.title("Impact of Fireplaces on SalePrice")
plt.plot(train.Fireplaces, train.SalePrice,'.', alpha = 0.3)




plt.xlabel("LotArea")
plt.ylabel("SalePrice($)")
plt.xlim(0,30000)
plt.title("Impact of LotArea on log_SalePrice")
plt.plot(train.LotArea, train.SalePrice,'.', alpha = 0.3)




train.dtypes.head(10) 




# Numerical variables

plt.hist(train.SalePrice, bins = 25)
plt.hist(np.log(train.SalePrice), bins = 25) # To make the distribution more symmetric




# Categorical variables:
# distribution of Foundation in our training set

train.Foundation.value_counts()




sns.countplot(train.Foundation)
sns.countplot(test.Foundation)




# exploring the relationships between variables 

plt.plot(train.GrLivArea, train.SalePrice,
         '.', alpha = 0.3)
 
plt.plot(train.GrLivArea, np.log(train.SalePrice),
         '.', alpha = 0.3)




# There are two points that don’t seem to fit in with the rest ( outliers ). 
# we don’t want the model to learn from & they can simply be removed. 




plt.plot(train.YearBuilt, train.GarageYrBlt,
         '.', alpha=0.5, label = 'training set') # alpha : to make the points partially transparent. 
 
plt.plot(test.YearBuilt, test.GarageYrBlt,
         '.', alpha=0.5, label = 'test set')
 
plt.legend()




plt.plot(train.YearBuilt, np.log(train.SalePrice),
         '.', alpha = 0.3)




# Neighborhood vs SalePrice
neighbor_price = (train[["Neighborhood", "SalePrice"]].groupby(['Neighborhood'], as_index=False).mean().sort_values
      (by='SalePrice', ascending=False))

neighbor_price.head()




# Create a single dataframe: Join Train and Test Dataset
# In order not to repeat our cleaning steps twice (train and test)/ will separate again.

data = pd.concat((train, test)).reset_index(drop=True)
# or 
# data = train.append(test, sort=False).reset_index(drop=True)

print(train.shape, test.shape, data.shape)
# data.info()




data.head()




# Dropping features that showed high correlation
data.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd'],axis =1, inplace = True)




# filling NA's with the mean of the column:  

data = data.fillna(data.mean())
data.head()




# Non-numerical Features
data.dtypes.value_counts()




# Check missing values 

count_mv = data.isnull().sum().sort_values(ascending=False)
count_mv.head()

percent_mv = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
percent_mv.head()

missing_frame = pd.concat([count_mv, percent_mv], axis = 1)
missing_frame.head()




# Missing for a reason: some of the missing values are in fact meaningful: 
# For categorical features replacing with 'None'
# For numerical features replacing with '0' 




data = pd.get_dummies(data)
data.head() 




y = np.log(train.SalePrice)
X = data.drop(['SalePrice','Id'], axis=1) 




train_shape = train.shape[0]

X_train = data[:train_shape]
X_test = data[train_shape:]




#print("Accuracy : ", model_ridge.score(X_test, y_test)*100)




#from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=.33)




X_train.shape
#y.shape




X_train.dtypes.value_counts()




X_train.select_dtypes(include = [object]).columns




# RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
# Fit/Accurancy
rf_model = rf.fit(X_train,y) 

pred_r = rf_model.predict(X_test)

rf_predict = pd.DataFrame(dict(Id=test.Id, SalePrice=np.expm1(pred_r)))
rf_predict.head()




# Ridge

from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=22)
# Fit/Accurancy
ridge_model.fit(X_train, y)

# Prediction
pred_ri = ridge_model.predict(X_test)

ridge_predict = pd.DataFrame(dict(Id=test.Id, SalePrice=np.expm1(pred_ri)))
ridge_predict.head()




# Lasso

from sklearn.linear_model import Lasso

model_lasso = Lasso(alpha=0.00055)

# Fit/Accurancy
model_lasso.fit(X_train, y)

# Prediction
preds_l = model_lasso.predict(X_test)

lasso_preds = pd.DataFrame(dict(SalePrice=np.expm1(preds_l), Id=test.Id))
lasso_preds.head()




# Convert DataFrame to a csv file 

filename = 'House_Price_Predictions.csv'

rf_predict.to_csv('../'+filename,index=False)

print('Saved file: ' + filename) 







