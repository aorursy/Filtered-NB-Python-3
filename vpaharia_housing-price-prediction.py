#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')




housing = pd.read_csv('../input/train.csv')
housing_test = pd.read_csv('../input/test.csv')
#combining train and test data, so that data imputation can be performed on complete set
housing_combined = pd.concat([housing, housing_test], axis=0,ignore_index=True)



              
sb.boxplot(x='PoolQC', y='PoolArea', data=housing_combined)




housing_combined.loc[2420,'PoolQC'] = 'Ex'
housing_combined.loc[2503,'PoolQC'] = 'Ex'
housing_combined.loc[2599,'PoolQC'] = 'Fa'




housing_combined['Alley'] = housing_combined['Alley'].fillna('None')         
housing_combined['PoolQC'] = housing_combined['PoolQC'].fillna('None')
housing_combined['Fence'] = housing_combined['Fence'].fillna('None')
housing_combined['MiscFeature'] = housing_combined['MiscFeature'].fillna('None')
housing_combined['FireplaceQu'] = housing_combined['FireplaceQu'].fillna('None')




sb.countplot(x='Electrical', data=housing_combined)
housing_combined['Electrical'] = housing_combined['Electrical'].fillna('Sbrkr')




#Fill missing Garage year build with build year of house.
housing_combined['GarageYrBlt'] = housing_combined['GarageYrBlt'].fillna(housing_combined['YearBuilt'])




#mark this house without 'GarageCars' as without garage
housing_combined.loc[2576,'GarageCars'] = 0
housing_combined.loc[2576,'GarageArea'] = 0




#update missing values with most common once
print(housing_combined[(housing_combined['GarageFinish'].isnull() == True) & (housing_combined['GarageArea'] > 0)][['GarageArea','GarageFinish','GarageCars','GarageType','GarageQual','GarageCond']])
housing_combined.loc[2126,['GarageQual','GarageCond','GarageFinish']] = ['Gd','TA','RFn']




#"NA means not Present"
housing_combined['GarageType'] = housing_combined['GarageType'].fillna('None')  
housing_combined['GarageQual']= housing_combined['GarageQual'].fillna('None')
housing_combined['GarageCond']  = housing_combined['GarageCond'].fillna('None')
housing_combined['GarageFinish'] = housing_combined['GarageFinish'].fillna('None')





sb.countplot(housing_combined['MasVnrType'])
housing_combined.loc[2610,'MasVnrType'] = 'BrkCmn'

housing_combined['MasVnrType'] = housing_combined['MasVnrType'].fillna('None')
housing_combined['MasVnrArea'] = housing_combined['MasVnrArea'].fillna(0)




sb.distplot(housing_combined['LotFrontage'].dropna())

lotFrontageByNeighborhood = housing_combined.groupby(['Neighborhood'])['LotFrontage'].mean()
import math

housing_combined['LotFrontage'] = housing_combined.apply(lambda row:lotFrontageByNeighborhood[row['Neighborhood']] if math.isnan(row['LotFrontage'])                                 else row['LotFrontage'], axis=1)




fig, axs = plt.subplots(ncols=4,figsize=(15,5))
sb.countplot(x='Functional', data=housing_combined, ax=axs[0])
sb.countplot(x='KitchenQual', data=housing_combined, ax=axs[1])
sb.countplot(x='SaleType', data=housing_combined, ax=axs[2])
sb.countplot(x='Utilities', data=housing_combined, ax=axs[3])




housing_combined['Functional'].fillna('Typ', inplace=True)
housing_combined['KitchenQual'].fillna('TA', inplace=True)
housing_combined['SaleType'].fillna('WD', inplace=True)
housing_combined.drop(['Utilities'], axis=1, inplace=True)




fig, axs = plt.subplots(ncols=2,figsize=(15,5))
sb.countplot(x='Exterior1st', data=housing_combined, ax=axs[0])
sb.countplot(x='Exterior2nd', data=housing_combined, ax=axs[1])




housing_combined['Exterior1st'].fillna('Other', inplace=True)
housing_combined['Exterior2nd'].fillna('Other', inplace=True)




print(housing_combined[housing_combined['MSZoning'].isnull() == True][['MSZoning', 'MSSubClass']])
sb.countplot(housing_combined['MSSubClass'],hue=housing_combined['MSZoning'])

housing_combined.loc[[1915,2250],'MSZoning'] = 'RM'
housing_combined.loc[[2216,2904],'MSZoning'] = 'RL'




bsmtCols = housing_combined.columns[housing_combined.columns.str.startswith('Bsmt')]

#missing surface area means no surface area i.e. does not exists in house
housing_combined['BsmtUnfSF'] = housing_combined['BsmtUnfSF'].fillna(0)
housing_combined['TotalBsmtSF'] = housing_combined['TotalBsmtSF'].fillna(0)
housing_combined['BsmtFinSF1'] = housing_combined['BsmtFinSF1'].fillna(0)
housing_combined['BsmtFinSF2'] = housing_combined['BsmtFinSF2'].fillna(0)
housing_combined['BsmtFullBath'] = housing_combined['BsmtFullBath'].fillna(0)
housing_combined['BsmtHalfBath'] = housing_combined['BsmtHalfBath'].fillna(0)

#if surface are is 0 then basement of that type does not exists
housing_combined['BsmtFinType1'] = housing_combined.apply(lambda x : 'None' if x['BsmtFinSF1'] == 0 else x['BsmtFinType1'],axis=1)
housing_combined['BsmtFinType2'] = housing_combined.apply(lambda x : 'None' if x['BsmtFinSF2'] == 0 else x['BsmtFinType2'],axis=1)

#if any of the surface are exists then 'BsmtExposure' is NO otherwise 'None'
housing_combined['BsmtExposure'] = housing_combined.apply(lambda x : 'NO' if x['TotalBsmtSF'] > 0 or x['BsmtFinSF2'] > 0 or x['BsmtFinSF1'] > 0 or x['BsmtUnfSF'] > 0 else 'None',axis=1)

#if no surface are of any type then 'no basement i.e. does not exists in house
housing_combined['BsmtCond'] = housing_combined.apply(lambda x : x['BsmtCond'] if x['TotalBsmtSF'] > 0 or x['BsmtFinSF2'] > 0 or x['BsmtFinSF1'] > 0 or x['BsmtUnfSF'] > 0 else 'None',axis=1)
housing_combined['BsmtQual'] = housing_combined.apply(lambda x : x['BsmtQual'] if x['TotalBsmtSF'] > 0 or x['BsmtFinSF2'] > 0 or x['BsmtFinSF1'] > 0 or x['BsmtUnfSF'] > 0 else 'None',axis=1)




#few more missing values
housing_combined[housing_combined[bsmtCols].isnull().any(axis=1)][bsmtCols]




#fill 'BsmtFinType2' based on its surface area
plt.figure(figsize=(10,12))
sb.boxplot(x='BsmtFinType2',y='BsmtFinSF2', data=housing_combined)

housing_combined.loc[332,'BsmtFinType2'] = 'ALQ'




#fill based on Total surface area
plt.figure(figsize=(10,12))
sb.boxplot(x='BsmtQual',y='TotalBsmtSF', data=housing_combined)

housing_combined.loc[2217,'BsmtQual'] = 'Fa'
housing_combined.loc[2218,'BsmtQual'] =  'Fa'




#filled based on most common
sb.countplot(x='BsmtCond', data=housing_combined)

housing_combined.loc[2040,'BsmtCond'] = 'Gd'
housing_combined.loc[2185,'BsmtCond'] = 'Gd'
housing_combined.loc[2524,'BsmtCond'] = 'Po'




#all columns have no null values
housing_combined.info()




areaCols = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',            '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',            '3SsnPorch', 'ScreenPorch', 'PoolArea']




#total buiding surface area
housing_combined['BldSF'] = housing_combined[['TotalBsmtSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF'                              , 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']]                    .sum(axis=1)




#total construction area
housing_combined['TotalSF'] = housing_combined[['BldSF', 'PoolArea', 'LotFrontage']].sum(axis=1)




sb.countplot(x='MoSold', data=housing_combined)




housing_combined['HighSeason'] = housing_combined['MoSold'].apply(lambda x: int(x in [5,6,7]))




sb.countplot(x='YrSold', data=housing_combined)




housing_combined['LowYear'] = housing_combined['YrSold'].apply(lambda x: int(x == 2010))




plt.figure(figsize=(20,12))
sb.boxplot(x='Neighborhood', y='SalePrice', data=housing_combined)




housing_combined['RichNeighborhood'] = housing_combined['Neighborhood'].apply(lambda x: int(x in ['NoRidge', 'NridgHt', 'StoneBr']))




neighborhoodByMedianSalePrice = housing_combined.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
plt.figure(figsize=(10,15))
neighborhoodByMedianSalePrice.plot(kind='bar')




mapNeighborhoodBySalePrice = {'MeadowV' : 0, 'IDOTRR' : 1, 'BrDale' : 1, 'OldTown' : 2, 'Edwards' : 2,              'BrkSide' : 2, 'Sawyer' : 3, 'Blueste' : 3, 'SWISU' : 3, 'NAmes' : 3, 'NPkVill' : 4, 'Mitchel' : 4,              'SawyerW' : 5, 'Gilbert' : 5, 'NWAmes' : 5, 'Blmngtn' : 5, 'CollgCr' : 5, 'ClearCr' : 5,              'Crawfor' : 5, 'Veenker' : 6, 'Somerst' : 6, 'Timber' : 6, 'StoneBr' : 7, 'NoRidge' : 7,              'NridgHt' : 7}




housing_combined['NeighborhoodBins'] = housing_combined['Neighborhood'].map(mapNeighborhoodBySalePrice)




housing_combined['NewHouse'] = (housing_combined['YearBuilt'] == housing_combined['YearRemodAdd']).astype(int)
housing_combined['Age'] = housing_combined['YrSold'] - housing_combined['YearBuilt']
housing_combined['AgeAfterRemodAdd'] = housing_combined['YrSold'] - housing_combined['YearRemodAdd']




plt.figure(figsize=(15,10))
sb.boxplot(x='SaleCondition', y='SalePrice', data=housing_combined, hue='OverallQual')




#Abnormally sold house are sold for short. Same quality houses are sold for less
housing_combined['ShortSale'] = (housing_combined['SaleCondition'] == 'Abnorml').astype(int)




plt.figure(figsize=(15,10))
sb.boxplot(x='SaleType', y='SalePrice', data=housing_combined)




sb.countplot(x='SaleType', data=housing_combined)




#pool is only in train data, no elevator, garage2 and other very few values
housing_combined['Shed'] = (housing_combined['MiscFeature'] == 'Shed').astype(int)




qualityCodeMap = {"Ex" : 0, "Gd" :1 , "TA" :2 ,"Fa" :3 ,"Po" :4 ,"None" : 5}
housing_combined['HeatingQC'] = housing_combined['HeatingQC'].map(qualityCodeMap)
housing_combined['ExterQual'] = housing_combined['ExterQual'].map(qualityCodeMap)
housing_combined['ExterCond'] = housing_combined['ExterCond'].map(qualityCodeMap)
housing_combined['BsmtCond'] = housing_combined['BsmtCond'].map(qualityCodeMap)
housing_combined['BsmtQual'] = housing_combined['BsmtQual'].map(qualityCodeMap)
housing_combined['KitchenQual'] = housing_combined['KitchenQual'].map(qualityCodeMap)
housing_combined['FireplaceQu'] = housing_combined['FireplaceQu'].map(qualityCodeMap)
housing_combined['GarageQual'] = housing_combined['GarageQual'].map(qualityCodeMap)
housing_combined['GarageCond'] = housing_combined['GarageCond'].map(qualityCodeMap)
housing_combined['PoolQC'] = housing_combined['PoolQC'].map(qualityCodeMap)




housing_combined.drop(['MoSold', 'YrSold', 'Neighborhood', 'YearBuilt', 'YearRemodAdd', 'YrSold',                'SaleCondition', 'SaleType', 'MiscFeature'], axis=1, inplace=True)




objectColums = housing_combined.dtypes[housing_combined.dtypes == object]
objectColumnDummies = pd.get_dummies(housing_combined[objectColums.index],drop_first=True)




objectColumnDummies.info()




housing_combined = pd.concat([housing_combined,objectColumnDummies],axis=1)




housing_combined.drop(objectColums.index,inplace=True,axis=1)




skew_colums = housing_combined.drop('SalePrice',axis=1).apply(lambda x : x.skew())
skew_colums = skew_colums[skew_colums > 0.75]




housing_combined[skew_colums.index] = housing_combined[skew_colums.index].apply(lambda x: np.log1p(x))




housing_train = housing_combined[:1460]
housing_test = housing_combined[1460:].drop('SalePrice',axis=1)




sb.distplot(housing_train['SalePrice'])




housing_train['SalePrice'] = np.log1p(housing_train['SalePrice'])




sb.distplot(housing_train['SalePrice'])




from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler

param_grid = {'alpha' : [0.0001,0.001,0.01,0.1,1,10,100,1000]}

print('Best score: {}'.format(pipelineRidge.steps[1][1].best_score_))
print('Best parameters: {}'.format(pipelineRidge.steps[1][1].best_params_))




predictRidge = pipelineRidge.predict(housing_test)
predictRidge = np.expm1(predictRidge)
predictRidgedf = pd.DataFrame(predictRidge,columns=['SalePrice'],index=housing_test['Id'])
predictRidgedf.to_csv('TestRidge.csv')






