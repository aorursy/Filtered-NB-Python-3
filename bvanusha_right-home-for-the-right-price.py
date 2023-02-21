#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # for data processing,  read CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # pip install --upgrade numpy scipy pandas
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV




# Load the data
all_data = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_labels = all_data["SalePrice"]
#print all_data.shape
#print test_df.shape




isPlots = "false"

if(isPlots=="true"):
    print (all_data.describe())




# check how many missing values each column has
def missing_values(df):
    for feature in list(df.columns.values):
        if len(all_data[feature])-len(all_data[feature].dropna())!=0:
            print (feature, ": ", len(all_data[feature])-len(all_data[feature].dropna()), "/", len(all_data[feature]))
        
missing_values(all_data)        




# we can ignore the features which have >70% (~1022) of missing values
#all_data.loc[all_data.Alley.isnull(), 'Alley'] = 'NoAlley'

del all_data['Alley']
del all_data['PoolQC']
del all_data['Fence']
del all_data['MiscFeature']




# plot house price
if(isPlots=="true"):
    sns.distplot(all_data['SalePrice'], kde = True, color = 'b') # http://seaborn2.readthedocs.io/en/latest/generated/seaborn.distplot.html




#for feature in list(all_data.columns.values):
#    print (feature, all_data[feature].unique())




if(isPlots=="true"):
    print ("Statistics:")
    print (all_data["SalePrice"].describe())
    print ("\nThe mean of the House Price is: ", all_data['SalePrice'].mean())
    print ("The median of the House Price is: ", all_data['SalePrice'].median(axis = 0))
    print (list(all_data.select_dtypes(include = ['float64', 'int64']).columns.values))




## Numerical Features
corr = all_data.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
if(isPlots=="true"):
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, vmax=1, square=False)




# build corr_dict of all the numerical features with correaltion to sale price
cor_dict = corr['SalePrice'].to_dict()
del cor_dict['SalePrice'] # delete "SalePrice", as that it's not a feature

if(isPlots=="true"):
    print("List all the numerical features descendingly by their correlation with Sale Price:\n")
    for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
        print("{0}: \t{1}".format(*ele))




# let's see the correlation among selected features
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath',
           'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces']

corr = all_data[features].corr()
if(isPlots=="true"):
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, vmax=1, square=False)




if(isPlots=="true"):
    fig = plt.figure(2, figsize=(9, 7))
    plt.subplot(211)
    plt.scatter(all_data.YearBuilt.values, price)
    plt.title('YearBuilt')
    plt.subplot(212)
    plt.scatter(all_data.YearRemodAdd.values, price)
    plt.title('YearRemodAdd')
    fig.text(-0.01, 0.5, 'Sale Price', va = 'center', rotation = 'vertical', fontsize = 14)
    plt.tight_layout()




# filling in the missing values
print(all_data.select_dtypes(include=['object']).columns.values)




# MSZoning
'''
  'A'='Agriculture',
  'C (all)'='Commercial',
  'FV'='Floating Village Residential',
  'I'='Industrial',
  'RH'='Residential High Density',
  'RL'='Residential Low Density',
  'RP'='Residential Low Density Park',
  'RM'='Residential Medium Density'
'''
all_data.loc[all_data['MSZoning'].isnull(), 'MSZoning'] = 'RL' # Residential Low Density

if(isPlots=="true"):
    print(all_data['MSZoning'].unique())




# Utilities
all_data.loc[all_data['Utilities'].isnull(), 'Utilities'] = 'Public'

if(isPlots=="true"):
    print(all_data['Utilities'].unique())




# MasVnrType
all_data.loc[all_data['MasVnrType'].isnull(), 'MasVnrType'] = 'None'

if(isPlots=="true"):
    print(all_data['MasVnrType'].unique())




# Bsmt
all_data.loc[all_data['BsmtQual'].isnull(), 'BsmtQual'] = 'NoBsmt'
all_data.loc[all_data['BsmtCond'].isnull(), 'BsmtCond'] = 'NoBsmt'
all_data.loc[all_data['BsmtExposure'].isnull(), 'BsmtExposure'] = 'NoBsmt'
all_data.loc[all_data['BsmtFinType1'].isnull(), 'BsmtFinType1'] = 'NoBsmt'
all_data.loc[all_data['BsmtFinType2'].isnull(), 'BsmtFinType2'] = 'NoBsmt'

if(isPlots=="true"):
    print(all_data['BsmtQual'].unique())
    print(all_data['BsmtCond'].unique())
    print(all_data['BsmtExposure'].unique())
    print(all_data['BsmtFinType1'].unique())
    print(all_data['BsmtFinType2'].unique())




# Garage
all_data.loc[all_data['GarageType'].isnull(), 'GarageType'] = 'NoGarage'
all_data.loc[all_data['GarageFinish'].isnull(), 'GarageFinish'] = 'NoGarage'
all_data.loc[all_data['GarageQual'].isnull(), 'GarageQual'] = 'NoGarage'
all_data.loc[all_data['GarageCond'].isnull(), 'GarageCond'] = 'NoGarage'

if(isPlots=="true"):
    print(all_data['GarageType'].unique())
    print(all_data['GarageFinish'].unique())
    print(all_data['GarageQual'].unique())
    print(all_data['GarageCond'].unique())




# FirePlace
all_data.loc[all_data['FireplaceQu'].isnull(), 'FireplaceQu'] = 'NoFireplace'

if(isPlots=="true"):
    print(all_data['FireplaceQu'].unique())




# selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
numerical_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'MasVnrArea', 
                     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',  'BsmtFullBath', 
                     'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
                     '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

selected_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'MasVnrArea', 
                     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',  'BsmtFullBath', 
                     'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                     'Fireplaces', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
                     '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold','MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                     'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                     'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                     'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                     'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
                     'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
                     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                     'SaleType', 'SaleCondition']
train_data = all_data[selected_features]

#print x_all_data.describe()
print(all_data.shape)
print(train_data.shape)




# fill NA's with the mean of that column
# Converts categorical variable into dummy/indicator variables
train_data = pd.get_dummies(train_data)
train_data = train_data.fillna(train_data[:all_data.shape[0]].mean())




print train_data.shape




#print "before normalization: ", all_data_selected.describe()

# zscore_normalization
for c in numerical_features:
    train_data[c] = (train_data[c] - train_data[c].mean()) / train_data[c].std()

#print "after normalization: ",all_data_selected.describe()
#print list(train_data.columns.values)




# But first, we log transform the target: (reason well explained in Alexandru's AWESOME Notebook)
train_labels = np.log1p(train_labels)

train_data = pd.get_dummies(train_data)
print(train_data.shape)




from sklearn.cross_validation import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train_data, train_labels, scoring="mean_squared_error", cv=10))
    return(rmse)




# Thanks to Alexandru for his detailed explanation about Regression models
## Lasso CV
# get model using LassoCV (Lasso linear model with iterative fitting along a regularization path)
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
model_lasso = LassoCV(alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75], selection='random', max_iter=15000).fit(train_data, train_labels)

res = rmse_cv(model_lasso)
print("LassoCV: ",res.min())

coef = pd.Series(model_lasso.coef_, index = train_data.columns)
#print "Coefficeints of LassoCV are, ", model_lasso.coef_
print("LassoCV has picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")




## Ridge
model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)

if(isPlots=="true"):
    cv_ridge.plot(title = "Validation - Just Do It")
    plt.xlabel("alpha")
    plt.ylabel("rmse")

print("LassoCV: ",cv_ridge.min())






