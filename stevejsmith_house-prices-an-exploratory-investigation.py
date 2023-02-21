#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# machine learning

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier





# Return Ordinals for categorical data
def getOrdinals(f1, f2, df):
    """
    Returns the medians as Ordinals
    """
    medians = df[[f1, f2]].groupby([f1], as_index=False).median().sort_values(by=f1, ascending=True)
    medians[f1+'Ords'] = round(medians[f2]/100000,1)
    medians = medians.drop([f2], axis=1)
    return medians

def getOrdinalMeans(f1, f2, df):
    means = df[[f1, f2]].groupby([f1], as_index=False).mean().sort_values(by=f1, ascending=True)
    means[f1+'Ords'] = round(means[f2]/100000,1)
    means = means.drop([f2], axis=1)
    return means

# Feature Operations
def replaceFeature(list_of_features):
    """
    Used to restore a Feature to the combined dataframe
    """
    fetch_df = pd.read_csv('../input/train.csv')
    fetch_df1 = pd.read_csv('../input/test.csv')   
    for feature in list_of_features:
        combined[0][feature] = fetch_df[feature]
        combined[1][feature] = fetch_df1[feature]
    return

def dropColumn(feature, combined_df):
    train_df = combined_df[0]
    test_df = combined_df[1]
    train_df = train_df.drop([feature], axis=1)
    test_df = test_df.drop([feature], axis=1)
    combined = [train_df, test_df]
    return combined

def applyOrdinals(ordinal_df, combined_df):
    """
    Adds the ordinals to the combined dataframes
    """
    # creating a dict
    index1 = ordinal_df.columns.values[0]
    index2 = ordinal_df.columns.values[1]
    val_dict = ordinal_df.set_index(index1).to_dict()
    # creating a column with values from those in the dict
    
    for dataset in combined_df:
        dataset[index2] = dataset[index1].apply(lambda x: val_dict[index2][x])
    
    return combined


def removeNans(feature, combined_df, replacement):
    for i in range(len(combined_df)):
        combined_df[i].loc[ combined_df[i][feature].isnull(), feature] = replacement
    return combined_df

#Functions to return information
def trainFeatureInfo(feature, df):
    missing_vals = 1460 - df[feature].size
    values = set(df[feature])
    return missing_vals, values
    


def testFeatureInfo(feature, df):
    missing_vals = 1459 - df[feature].size
    values = set(df[feature])
    return missing_vals, values

def infoCheck(feature, train_df, test_df):

    train_info = trainFeatureInfo(feature, train_df)

    test_info = testFeatureInfo(feature, test_df)

    return train_info[0], test_info[0], train_info[1] == test_info[1]




train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combined = [train_df, test_df]




print(combined[0].columns.values)




print(combined[1].columns.values)




combined[0].head()




combined[0].describe()




print("Housing price Statistics:\n")
print(combined[0]['SalePrice'].describe())
print("\nThe median of the Housing Price is: ", train_df['SalePrice'].median(axis = 0))




sns.distplot(train_df['SalePrice'], kde=False, rug=True)




features_to_correlate = train_df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(features_to_correlate, robust=True, square=True)




cor_dict = features_to_correlate['SalePrice'].to_dict()
del cor_dict['SalePrice']
print("The numerical features and their correlation with SalePrice:\n")
for element in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
    print("{0}: \t{1}".format(*element))




#========
# Looking at Neighborhoods

plt.figure(figsize = (12, 10))
g = sns.boxplot(x = 'Neighborhood', y = 'SalePrice', data = combined[0])
xt = plt.xticks(rotation=90)




ordinals = getOrdinals('Neighborhood', 'SalePrice', combined[0])
ordinals




combined = applyOrdinals(ordinals, combined)
combined[0][['NeighborhoodOrds','SalePrice']].corr(method='pearson')




combined = dropColumn('Neighborhood', combined)




# Looking at SaleType
plt.figure(figsize = (12, 10))
g = sns.boxplot(x = 'SaleType', y = 'SalePrice', data = combined[0])
xt = plt.xticks(rotation=45)




# Fetching info
print(infoCheck('SaleType', combined[0], combined[1]))

ordinals = getOrdinals('SaleType', 'SalePrice', combined[0])
ordinals




# Replacing NaNs with the median category
combined = removeNans('SaleType', combined, 'COD')
combined = applyOrdinals(ordinals, combined)

combined[0][['SaleTypeOrds','SalePrice']].corr(method='pearson')




# drop column now ords created
combined = dropColumn('SaleType', combined)




#  Looking at SaleCondition

plt.figure(figsize = (12, 10))
g = sns.boxplot(x = 'SaleCondition', y = 'SalePrice', data = combined[0])
xt = plt.xticks(rotation=45)




print(infoCheck('SaleCondition', combined[0], combined[1]))

ordinals = getOrdinals('SaleCondition', 'SalePrice', combined[0])
ordinals




# Replacing NaNs with the median category
combined = removeNans('SaleCondition', combined, 'Family')
combined = applyOrdinals(ordinals, combined)

combined[0][['SaleConditionOrds','SalePrice']].corr(method='pearson')




combined = dropColumn('SaleCondition', combined)




#  Looking at HouseStyle
plt.figure(figsize = (12, 10))
g = sns.boxplot(x = 'HouseStyle', y = 'SalePrice', data = combined[0])
xt = plt.xticks(rotation=45)




print(infoCheck('HouseStyle', combined[0], combined[1]))

ordinals = getOrdinals('HouseStyle', 'SalePrice', combined[0])
ordinals




# Replacing NaNs with the median category
combined = removeNans('HouseStyle', combined, 'SFoyer')
combined = applyOrdinals(ordinals, combined)

combined[0][['HouseStyleOrds','SalePrice']].corr(method='pearson')




to_drop = ['HouseStyleOrds', 'HouseStyle']
for feature in to_drop:
    combined = dropColumn(feature, combined)
 




# Housing Condition
# Condition1

plt.figure(figsize = (12, 10))
g = sns.boxplot(x = 'Condition1', y = 'SalePrice', data = combined[0])
xt = plt.xticks(rotation=45)




print(infoCheck('Condition1', combined[0], combined[1]))

ordinals1 = getOrdinals('Condition1', 'SalePrice', combined[0])
print(ordinals1)

ordinals2 = getOrdinalMeans('Condition1', 'SalePrice', combined[0])
print(ordinals2)




# Replacing NaNs with the median category
combined = removeNans('Condition1', combined, 'Norm')
combined = applyOrdinals(ordinals1, combined)

combined[0][['Condition1Ords','SalePrice']].corr(method='pearson')




combined = applyOrdinals(ordinals2, combined)

combined[0][['Condition1Ords','SalePrice']].corr(method='pearson')




to_drop = ['Condition1Ords', 'Condition1']
for feature in to_drop:
    combined = dropColumn(feature, combined)




# Exterior1st
plt.figure(figsize = (12, 10))
g = sns.boxplot(x = 'Exterior1st', y = 'SalePrice', data = combined[0])
xt = plt.xticks(rotation=45)




print(infoCheck('Exterior1st', combined[0], combined[1]))

ordinals = getOrdinals('Exterior1st', 'SalePrice', combined[0])
print(ordinals)




combined = removeNans('Exterior1st', combined, 'BrkFace')
combined = applyOrdinals(ordinals, combined)

combined[0][['Exterior1stOrds','SalePrice']].corr(method='pearson')




combined = dropColumn('Exterior1st', combined)




# Basement conditions

fig, ax = plt.subplots(2, 2, figsize = (10, 8))
sns.boxplot('BsmtCond', 'SalePrice', data = combined[0], ax = ax[0, 0])
sns.boxplot('BsmtQual', 'SalePrice', data = combined[0], ax = ax[0, 1])
sns.boxplot('BsmtExposure', 'SalePrice', data = combined[0], ax = ax[1, 0])
sns.boxplot('BsmtFinType1', 'SalePrice', data = combined[0], ax = ax[1, 1])




print(infoCheck('BsmtCond', combined[0], combined[1]))

print(infoCheck('BsmtQual', combined[0], combined[1]))

print(infoCheck('BsmtExposure', combined[0], combined[1]))

print(infoCheck('BsmtFinType1', combined[0], combined[1]))




ordinals1 = getOrdinals('BsmtCond', 'SalePrice', combined[0])
ordinals2 = getOrdinals('BsmtQual', 'SalePrice', combined[0])
ordinals3 = getOrdinals('BsmtExposure', 'SalePrice', combined[0])
ordinals4 = getOrdinals('BsmtFinType1', 'SalePrice', combined[0])

print(ordinals1)
print(ordinals2)
print(ordinals3)
print(ordinals4)




# Removing NaNs
combined = removeNans('BsmtCond', combined, 'Fa')
combined = removeNans('BsmtQual', combined, 'Gd')
combined = removeNans('BsmtExposure', combined, 'Av')
combined = removeNans('BsmtFinType1', combined, 'ALQ')





combined = applyOrdinals(ordinals1, combined)
combined = applyOrdinals(ordinals2, combined)
combined = applyOrdinals(ordinals3, combined)
combined = applyOrdinals(ordinals4, combined)
    
combined[0][['BsmtCondOrds','BsmtQualOrds', 'BsmtExposureOrds', 'BsmtFinType1Ords', 'SalePrice']].corr(method='pearson')




# combining some bsmtconds together


combined[0]['CombBsmtOrds'] = combined[0]['BsmtCondOrds'] + combined[0]['BsmtExposureOrds']         + combined[0]['BsmtFinType1Ords']
        
combined[0][['CombBsmtOrds', 'SalePrice']].corr(method='pearson')




for dataset in combined:
    dataset['CombBsmtOrds'] = combined[0]['BsmtCondOrds'] + combined[0]['BsmtExposureOrds']         + combined[0]['BsmtFinType1Ords']
        
combined = dropColumn('BsmtCondOrds', combined)
combined = dropColumn('BsmtExposureOrds', combined)
combined = dropColumn('BsmtFinType1Ords', combined)




to_drop =['BsmtCond', 'BsmtQual', 'BsmtExposure','BsmtFinType1']
for feature in to_drop:
    combined = dropColumn(feature, combined)




# Home Functionality

sns.violinplot('Functional', 'SalePrice', data = combined[0])




# doesn't look promising, let's try
ordinals = getOrdinals('Functional', 'SalePrice', combined[0])
ordinals




combined = removeNans('Functional', combined, 'Mod')
combine_test = applyOrdinals(ordinals, combined)

combine_test[0][['FunctionalOrds', 'SalePrice']].corr(method='pearson')




combined = dropColumn('Functional', combined) 
combined = dropColumn('FunctionalOrds', combined) 




#=============================

# Looking at FirePlaceQu

sns.factorplot('FireplaceQu', 'SalePrice', data = combined[0], color = 'm',                estimator = np.median, order = ['Ex', 'Gd', 'TA', 'Fa', 'Po'], size = 4.5,  aspect=1.35)




# looks very promising


ordinals = getOrdinals('FireplaceQu', 'SalePrice', combined[0])
ordinals




combined = removeNans('FireplaceQu', combined, 'TA')
combined = applyOrdinals(ordinals, combined)

combined[0][['FireplaceQuOrds', 'SalePrice']].corr(method='pearson')




combined = dropColumn('FireplaceQu', combined)




#===================

# looking at HeatingQC against saleprice and whether there is CentralAir info

sns.factorplot('HeatingQC', 'SalePrice', hue = 'CentralAir', estimator = np.mean, data = combined[0], 
             size = 4.5, aspect = 1.4)




# let's have a look at the CentralAir data


# Y for yes and N for no, changevalues
for dataset in combined:
    dataset['CentralAir'] = dataset['CentralAir'].map( {'Y': 2, 'N':1}).astype(int)
      
combined[0]['CentralAir'].head()




# let's create ordinals for HeatingQC


ordinals = getOrdinals('HeatingQC', 'SalePrice', combined[0])
ordinals




combined = removeNans('HeatingQC', combined, 'TA')
combined = applyOrdinals(ordinals, combined)

# Creating an artificial feature
for dataset in combined:
    dataset['CenAirHeQcOrds'] = dataset['HeatingQCOrds'] * dataset['CentralAir']
    

combined[0][['CenAirHeQcOrds','HeatingQCOrds','CentralAir'  , 'SalePrice']].corr(method='pearson')




# Keep CenAirHeQcOrds
combined = dropColumn('HeatingQCOrds', combined) 
combined = dropColumn('CentralAir', combined) 
combined = dropColumn('HeatingQC', combined)




# Electrical

sns.boxplot('Electrical', 'SalePrice', data = combined[0])




ordinals = getOrdinals('Electrical', 'SalePrice', combined[0])
ordinals




combined = removeNans('Electrical', combined, 'FuseA')
combined = applyOrdinals(ordinals, combined)

combined[0][['ElectricalOrds'  , 'SalePrice']].corr(method='pearson')




# Poor, let's drop

combined = dropColumn('ElectricalOrds', combined) 
combined = dropColumn('Electrical', combined)




#==============

# Kitchen Quality

sns.factorplot('KitchenQual', 'SalePrice', estimator = np.mean, 
               size = 4.5, aspect = 1.4, data = combined[0], order = ['Ex', 'Gd', 'TA', 'Fa'])




ordinals = getOrdinals('KitchenQual', 'SalePrice', combined[0])
ordinals




combined = removeNans('KitchenQual', combined, 'TA')
combined = applyOrdinals(ordinals, combined)

combined[0][['KitchenQualOrds'  , 'SalePrice']].corr(method='pearson')




# Keep KitchenQualOrds
combined = dropColumn('KitchenQual', combined)




#=================

# MSZoning

sns.boxplot(x = 'MSZoning', y = 'SalePrice', data = combined[0])




ordinals = getOrdinals('MSZoning', 'SalePrice', combined[0])
ordinals




combined = removeNans('MSZoning', combined, 'RH')

            
combined = applyOrdinals(ordinals, combined)

combined[0][['MSZoningOrds'  , 'SalePrice']].corr(method='pearson')




# Poor, delete

combined = dropColumn('MSZoning', combined)
combined = dropColumn('MSZoningOrds', combined)


#=====

# Street and Alley removed

combined = dropColumn('Street', combined)
combined = dropColumn('Alley', combined)

#=====

# Checking where we are

list(combined[0].select_dtypes(include=['object']).columns.values)




# GarageFeatures

garage_features = ['GarageType', 'GarageFinish', 'GarageQual',  'GarageCond',]

fig, ax = plt.subplots(4, 1, figsize = (10, 8))

i = 0
for feature in garage_features:
    
    sns.boxplot(x = feature, y = 'SalePrice', data = combined[0], ax=ax[i])
    i += 1




# NaN is No garage
combined = removeNans('GarageType', combined, 'None')
combined = removeNans('GarageFinish', combined, 'None')
combined = removeNans('GarageQual', combined, 'None')
combined = removeNans('GarageCond', combined, 'None')




ordinals = []
for feature in garage_features:
    ordinals.append(getOrdinals(feature, 'SalePrice', combined[0]))

for df in ordinals:
    combined = applyOrdinals(df, combined)

combined[0][['GarageTypeOrds', 'GarageFinishOrds', 'GarageQualOrds',  'GarageCondOrds', 'SalePrice']].corr(method='pearson') 




# All Garage features are correlated, let's make a single feature

for dataset in combined:
    dataset['CombGaraOrds'] = dataset['GarageTypeOrds'] * dataset['GarageFinishOrds']         * dataset['GarageQualOrds'] * dataset['GarageCondOrds']

combined[0][['CombGaraOrds', 'SalePrice']].corr(method='pearson') 




# Keep this, drop the rest
to_drop = ['GarageType', 'GarageFinish', 'GarageQual',  'GarageCond', 'GarageTypeOrds', 'GarageFinishOrds', 'GarageQualOrds',  'GarageCondOrds']

for feature in to_drop:
    combined = dropColumn(feature, combined)




# Exterior features

exterior_features = [ 'MasVnrType', 'ExterQual', 'ExterCond']

fig, ax = plt.subplots(3, 1, figsize = (10, 8))

i = 0
for feature in exterior_features:
    
    sns.boxplot(x = feature, y = 'SalePrice', data = combined[0], ax=ax[i])
    i += 1




ordinals = []
for feature in exterior_features:
    ordinals.append(getOrdinals(feature, 'SalePrice', combined[0]))
    
combined = removeNans('MasVnrType', combined, 'None')
combined = removeNans('ExterQual', combined, 'TA')
combined = removeNans('ExterCond', combined, 'Gd')

for df in ordinals:
    combined = applyOrdinals(df, combined)

combined[0][['MasVnrTypeOrds', 'ExterQualOrds', 'ExterCondOrds', 'SalePrice']].corr(method='pearson') 




# Let's combine MasVnrTypeOrds & ExterQualOrds

for dataset in combined:
    dataset['CombExtOrds'] = dataset['MasVnrTypeOrds'] + dataset['ExterQualOrds'] 
    

combined[0][['CombExtOrds', 'SalePrice']].corr(method='pearson')




# Worse, drop this column and keep them seperate

to_drop = ['MasVnrType', 'ExterQual', 'ExterCond', 'ExterCondOrds', 'CombExtOrds']

for feature in to_drop:
    combined = dropColumn(feature, combined)




#======

# Utilities

sns.factorplot('Utilities', 'SalePrice', estimator = np.mean, data = combined[0], 
             size = 4.5, aspect = 1.4)




ordinals = getOrdinalMeans('Utilities', 'SalePrice', combined[0])
ordinals




# too close for anything meaningful

combined = dropColumn('Utilities', combined)




shape_features = ['LotShape',  'LandContour', 'LotConfig', 'LandSlope']

fig, ax = plt.subplots(4, 1, figsize = (10, 8))

i = 0
for feature in shape_features:
    
    sns.boxplot(x = feature, y = 'SalePrice', data = combined[0], ax=ax[i])
    i += 1




# Nothing promising, delete

for feature in shape_features:
    combined = dropColumn(feature, combined)




# drop 2nd features

second_features = ['Condition2', 'Exterior2nd',  'BsmtFinType2']

for feature in second_features:
    combined = dropColumn(feature, combined)




#====
# Checking where we are

list(combined[0].select_dtypes(include=['object']).columns.values)





# BldgType

sns.factorplot('BldgType', 'SalePrice', estimator = np.mean, data = combined[0], 
             size = 4.5, aspect = 1.4)




ordinals = getOrdinalMeans('BldgType', 'SalePrice', combined[0])
ordinals




combined = removeNans('BldgType', combined, 'Twnhs')


combined = applyOrdinals(ordinals, combined)

combined[0][['BldgTypeOrds', 'SalePrice']].corr(method='pearson') 




# poor, delete
combined = dropColumn('BldgTypeOrds', combined)
combined = dropColumn('BldgType', combined)




remaining_features = list(combined[0].select_dtypes(include=['object']).columns.values)




combined = removeNans('RoofStyle', combined, 'Gable')
combined = removeNans('RoofMatl', combined, 'Tar&Grv')
combined = removeNans('Foundation', combined, 'CBlock')
combined = removeNans('Heating', combined, 'Wall')
combined = removeNans('PavedDrive', combined, 'N')
combined = removeNans('PoolQC', combined, 'None')
combined = removeNans('Fence', combined, 'None')
combined = removeNans('MiscFeature', combined, 'None')

ordinals = []
for feature in remaining_features:
    ordinals.append(getOrdinals(feature, 'SalePrice', combined[0]))
    
import pprint
pprint.pprint(ordinals)




for df in ordinals:
    combined = applyOrdinals(df, combined)

combined[0][['RoofStyleOrds',
        'RoofMatlOrds',
        'FoundationOrds',
        'HeatingOrds',
        'PavedDriveOrds',
        'PoolQCOrds',
        'FenceOrds',
        'MiscFeatureOrds', 'SalePrice']].corr(method='pearson') 




# keep FoundationOrds, nothing else worth keeping

to_drop = ['RoofStyleOrds', 'RoofMatlOrds', 'HeatingOrds',
        'PavedDriveOrds', 'PoolQCOrds', 'FenceOrds',
        'MiscFeatureOrds', 'RoofStyle', 'RoofMatl',
        'Foundation', 'Heating', 'PavedDrive', 'PoolQC',
        'Fence', 'MiscFeature']

for feature in to_drop:
    combined = dropColumn(feature, combined)

# check    
list(combined[0].select_dtypes(include=['object']).columns.values)




# Seeing where we are with a correlation matrix

features_to_correlate = combined[0].select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(features_to_correlate, robust=True, square=True)




to_drop = ['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

for feature in to_drop:
    combined = dropColumn(feature, combined)

to_drop = ['MSSubClass', 'OverallCond', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr']
for feature in to_drop:
    combined = dropColumn(feature, combined)


features_to_correlate = combined[0].select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(features_to_correlate, robust=True, square=True)




# Combining 'LotArea' and 'LotFrontage'

for dataset in combined:
    dataset['LotCombined'] = dataset['LotArea'] + dataset['LotFrontage']
    
combined[0][['LotCombined',
        'LotArea',
        'LotFrontage', 'SalePrice']].corr(method='pearson') 




# makes it worse, delete LotArea and LotCombined

to_drop = ['LotArea', 'LotCombined']
for feature in to_drop:
    combined = dropColumn(feature, combined)




# Look TotRmsAbvGrd BedroomAbvGr
# Combining 'LotArea' and 'LotFrontage'

for dataset in combined:
    dataset['RmsAbvGrd'] = dataset['TotRmsAbvGrd'] + dataset['BedroomAbvGr']
    
combined[0][['RmsAbvGrd',
        'TotRmsAbvGrd',
        'BedroomAbvGr', 'SalePrice']].corr(method='pearson') 




# Remove RmsAbvGrd, BedroomAbvG

to_drop = ['RmsAbvGrd', 'BedroomAbvGr']
for feature in to_drop:
    combined = dropColumn(feature, combined)
    
    
# Combining 'WoodDeckSF' and 'OpenPorchSF'

for dataset in combined:
    dataset['PorchDeck'] = dataset['WoodDeckSF'] + dataset['OpenPorchSF']
    
combined[0][['PorchDeck',
        'WoodDeckSF',
        'OpenPorchSF', 'SalePrice']].corr(method='pearson')  




# PorchDeck is an inprovement
to_drop = ['WoodDeckSF', 'OpenPorchSF']
for feature in to_drop:
    combined = dropColumn(feature, combined)


# checking where we are
features_to_correlate = combined[0].select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(features_to_correlate, robust=True, square=True)




# Combining 'FireplaceQuOrds', 'HalfBath',  BsmtFullBath'

for dataset in combined:
    dataset['FPBth'] = dataset['FireplaceQuOrds'] +            dataset['HalfBath'] + dataset['BsmtFullBath']
    
combined[0][['FPBth',
        'FireplaceQuOrds',
        'HalfBath', 'BsmtFullBath', 'SalePrice']].corr(method='pearson')




# improvement

to_drop = ['FireplaceQuOrds', 'HalfBath', 'BsmtFullBath']
for feature in to_drop:
    combined = dropColumn(feature, combined)




# Finding null vals and collating features

combined[0].info()




# MasVnrArea


# * null values, several NAs, replace NA with Zeros

for dataset in combined:
    dataset['MasVnrArea'].replace('NA', 0)

# replace null values with zero

combined = removeNans('MasVnrArea', combined, 0)

# GarageYrBlt
# There's a value of 2207

for dataset in combined:
    dataset['GarageYrBlt'].replace(2207, 2007)




# replace null values with 1980

combined = removeNans('GarageYrBlt', combined, 1980)

# checking correlation
combined[0][['GarageYrBlt','SalePrice']].corr(method='pearson')




# LotFrontage          1201 non-null float64

combined[0]['LotFrontage'].median()




# replace null values with 69

combined = removeNans('LotFrontage', combined, 69)

combined[0][['LotFrontage','SalePrice']].corr(method='pearson')




combined[1].info()




# looking at data, null should be zeros

combined = removeNans('BsmtFinSF1', combined, 0)
combined = removeNans('BsmtUnfSF', combined, 0)
combined = removeNans('TotalBsmtSF', combined, 0)
combined = removeNans('GarageCars', combined, 0)
combined = removeNans('GarageArea', combined, 0)
combined = removeNans('FPBth', combined, combined[0]['FPBth'].median())

combined[0][['BsmtFinSF1',
        'BsmtUnfSF', 
        'TotalBsmtSF', 'SalePrice']].corr(method='pearson')




# there doesn't need to be 3 basement square feet variables, would provide
# skew
to_drop = ['BsmtFinSF1', 'BsmtUnfSF']
for feature in to_drop:
    combined = dropColumn(feature, combined)




# attempting to reduce garage features

combined[0][['GarageYrBlt',
        'GarageCars', 
        'GarageArea', 'SalePrice']].corr(method='pearson')




for dataset in combined:
    dataset['GarageFeat'] = dataset['GarageYrBlt'] * dataset['GarageCars']             * dataset['GarageArea']
 
combined[0][['GarageFeat','SalePrice']].corr(method='pearson')




to_drop = ['GarageYrBlt', 'GarageCars', 'GarageArea']
for feature in to_drop:
    combined = dropColumn(feature, combined)




# combine Exterior Conds
# Combine Basement Quality

combined[0][['Exterior1stOrds', 'MasVnrTypeOrds',         'ExterQualOrds','SalePrice']].corr(method='pearson')




for dataset in combined:
    dataset['ExteriorConds'] = dataset['Exterior1stOrds'] +            dataset['MasVnrTypeOrds'] + dataset['ExterQualOrds']
           
combined[0][['ExteriorConds','SalePrice']].corr(method='pearson')




# keep ExterQualOrds
to_drop = ['Exterior1stOrds', 'MasVnrTypeOrds', 'ExteriorConds']
for feature in to_drop:
    combined = dropColumn(feature, combined)




# Combine Basement Quality

combined[0][['BsmtQualOrds', 'CombBsmtOrds','SalePrice']].corr(method='pearson')




for dataset in combined:
    dataset['BsmtConds'] = dataset['BsmtQualOrds'] +            dataset['CombBsmtOrds']
           
combined[0][['BsmtConds','SalePrice']].corr(method='pearson')




# improvement, drop BsmtQualOrds, CombBsmtOrds 

to_drop = ['BsmtQualOrds', 'CombBsmtOrds']
for feature in to_drop:
    combined = dropColumn(feature, combined)




# Preparing for Classifiers

train_df = combined[0]

train_df = train_df.drop('Id', axis=1)


test_df = combined[1]
   
combined = [train_df, test_df]




# split the data

X_train = combined[0].drop("SalePrice", axis=1)
Y_train = combined[0]["SalePrice"]

X_test = combined[1].drop("Id", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape




combined[1].info()




combined[0].info()




#======
# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train)*100,2)

acc_svc




# k-Nearest Neighbours algorithm

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train)*100, 2)
acc_knn




# Gaussain Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)
acc_gaussian




## Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train)*100, 2)
acc_perceptron




# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train)*100, 2)
acc_linear_svc




# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train)*100, 2)
acc_sgd




# # Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100, 2)
acc_decision_tree




# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100, 2)
acc_random_forest




models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN',  
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn,  
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)




# Using decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train)*100, 2)
acc_decision_tree


submission = pd.DataFrame({
        "Id": combined[1]['Id'],
        "SalePrice": Y_pred
        })
submission.head()




submission.tail()




#submission.to_csv('../output/submission3.csv', index=False)

