#!/usr/bin/env python
# coding: utf-8



#Import basic packages
import pandas as pd
import numpy as np
#Import packages for preprocessing (data cleaning)
from sklearn.preprocessing import RobustScaler, LabelEncoder
#Import packages for data visualisation
import matplotlib.pyplot as plt
#Import packages for model testing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
#Import packages for modeling
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline




#Import both training & testing set into Python. 
df_train_ori = pd.read_csv('../input/train.csv')
df_test_ori = pd.read_csv('../input/test.csv')

#Print the shape of both training & testing set. It makes sense as the target variable (SalePrice) is only available in training set.
print(df_train_ori.shape)
print(df_test_ori.shape)




df_test_ori['SalePrice'] = 0




#In this project, as we are predicting the SalePrice of a property, it is a common sense that there should be a linear relationsip
#between the area of the property and the sale price of the property. (The area of the property is in cloumn 'GrLivArea')

#We will use scatter plot to see if the relationship is linear. 
plt.scatter(x = df_train_ori['GrLivArea'], y = df_train_ori['SalePrice'])
plt.ylabel('SalePrice')
plt.xlabel('GrLivArea')
plt.show()




df_train_ori = df_train_ori.drop(df_train_ori[(df_train_ori['GrLivArea']>4000) & (df_train_ori['SalePrice']<300000)].index)




#Double check the shape of training set, in case we drop more points than expected. 
print(df_train_ori.shape)
#The result is as expected, only two points in the bottom right are dropped. 




#Combine the training&testing set
df_all = df_train_ori.append(df_test_ori)
#Double check the shape with the combined dataset. 
print(df_all.shape)




#Check the info for the combined dataset. 
print(df_all.info())




#By looking at the training & testing set, the missing value in these columns should be filled with 'None'. 
for i in ['Alley', 'MasVnrType','BsmtQual','BsmtCond','FireplaceQu', 'GarageType','GarageFinish','GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
    df_all[i] = df_all[i].fillna('None')




#The missing value in these columns should be filled with 0.
for i in ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','GarageYrBlt', 'GarageCars','GarageArea']:
    df_all[i] = df_all[i].fillna(0)




#The missing value in these columns should be filled with the mean of its corresponding feature.
df_all["LotFrontage"] = df_all["LotFrontage"].fillna(df_all["LotFrontage"].mean())




#The missing value in these columns should be filled with the most frequent number of its corresponding feature.
for i in ['MSZoning', 'Exterior1st','Exterior2nd','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'Electrical','BsmtFullBath', 'BsmtHalfBath','KitchenQual', 'Functional','SaleType']:
    df_all[i] = df_all[i].fillna(df_all[i].mode()[0])




#The missing value in these columns should be filled with the most frequent category of its corresponding feature.
df_all["Functional"] = df_all["Functional"].fillna("Typ")




df_all = df_all.drop(['Utilities'], axis = 1)




df_all['TotalAreaSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']




print(df_all.info())




#These variables are classified as categorical Variables with priority ranking
encode = ['LotShape','LandSlope','Neighborhood', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
          'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 
          'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
#Use encode() method to assign different numeric values to each of the category in each of the variable, 
for i in encode:
    le = LabelEncoder() 
    le.fit(list(df_all[i].values)) 
    df_all[i] = le.transform(list(df_all[i].values))




#These variables are classified as numeric variables
norm = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond','YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
        'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'TotalAreaSF']

#Combine encode & norm to normalize all the numeric variables
norm_variables = norm + encode
df_all[norm_variables] = (df_all[norm_variables]-df_all[norm_variables].mean())/(df_all[norm_variables].max()-df_all[norm_variables].min())




#These variables are classified as categorical Variables with equivalent ranking
cat = ['MSZoning', 'Street', 'Alley','LandContour', 'LotConfig','Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 'MiscFeature','SaleType',
       'SaleCondition']

df_all_dummy = pd.get_dummies(df_all, drop_first = True)
#This is the end of data cleaning process




df_train_adj = df_all_dummy[df_all_dummy['SalePrice'] != 0]
df_test_adj = df_all_dummy[df_all_dummy['SalePrice'] == 0]




#Training the data
data_to_train = df_train_adj.drop(['SalePrice','Id'], axis = 1)




df_train_adj["SalePrice"] = np.log1p(df_train_adj["SalePrice"])
labels_to_use = df_train_adj['SalePrice']




#Build and fit the model
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0007, random_state=1))
ridge = make_pipeline(RobustScaler(), Ridge(alpha =20, random_state=42))




def evaluation(model):
    result= np.sqrt(-cross_val_score(model, data_to_train, labels_to_use, cv = 5, scoring = 'neg_mean_squared_error'))
    return(result)




score = evaluation(lasso)
print("Lasso score: {:.5f}\n".format(score.mean()))




score = evaluation(ridge)
print("Ridge score: {:.5f} \n".format(score.mean()))




test_df_id = df_test_ori['Id']
test_df_x = df_test_adj.drop(['SalePrice', 'Id'], axis = 1)

lasso.fit(data_to_train, labels_to_use)
test_df_y_log = lasso.predict(test_df_x)
test_df_y = np.exp(1)**test_df_y_log

#Submission
submission = pd.DataFrame({'ID': list(test_df_id), 'SalePrice': list(test_df_y)})
submission.to_csv('submission.csv')

