#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, learning_curve, GridSearchCV
from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingClassifier

get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




print('Train: ('+str(train.shape[0])+','+str(train.shape[1])+')')
print('Test: ('+str(test.shape[0])+','+str(test.shape[1])+')')




train.head()




# Combine train/test datasets, also Drop Id(first column) column and target(last column, SalePrice)
data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                  test.loc[:,'MSSubClass':'SaleCondition']))
data.dtypes.value_counts()
y = train.SalePrice




numeric_features = data.select_dtypes(exclude = ["object"]).columns
print (numeric_features)




# Plot MSSubClass which is a nominal variable
sns.countplot(x="MSSubClass",data=data)




# Transform MSSubClass to an object
data.loc[:,'MSSubClass'] = data.loc[:,'MSSubClass'].astype(str)




categorical_features = data.select_dtypes(include = ["object"]).columns
print (categorical_features)




# Encode the categorical features as ordered numbers when there is information in the order
data = data.replace({"Alley" : {"Grvl" : 0, "Pave" : 1},
                     "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                     "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                     "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                     "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                     "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                     "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                     "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6,
                                     "Min1" : 7, "Typ" : 8},
                     "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                     "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                     "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                     "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                     "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                     "Street" : {"Grvl" : 1, "Pave" : 2},
                     "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}})




# Create new features
# 1* Simplifications of existing features
data["SimplOverallQual"] = data.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                     4 : 2, 5 : 2, 6 : 2, # average
                                                     7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                    })
data["SimplOverallCond"] = data.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                     4 : 2, 5 : 2, 6 : 2, # average
                                                     7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                    })
data["SimplPoolQC"] = data.PoolQC.replace({1 : 1, 2 : 1, # average
                                           3 : 2, 4 : 2 # good
                                          })
data["SimplGarageCond"] = data.GarageCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
data["SimplGarageQual"] = data.GarageQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
data["SimplFireplaceQu"] = data.FireplaceQu.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
data["SimplFireplaceQu"] = data.FireplaceQu.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
data["SimplFunctional"] = data.Functional.replace({1 : 1, 2 : 1, # bad
                                                   3 : 2, 4 : 2, # major
                                                   5 : 3, 6 : 3, 7 : 3, # minor
                                                   8 : 4 # typical
                                                  })
data["SimplKitchenQual"] = data.KitchenQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })
data["SimplHeatingQC"] = data.HeatingQC.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
data["SimplBsmtFinType1"] = data.BsmtFinType1.replace({1 : 1, # unfinished
                                                       2 : 1, 3 : 1, # rec room
                                                       4 : 2, 5 : 2, 6 : 2 # living quarters
                                                      })
data["SimplBsmtFinType2"] = data.BsmtFinType2.replace({1 : 1, # unfinished
                                                       2 : 1, 3 : 1, # rec room
                                                       4 : 2, 5 : 2, 6 : 2 # living quarters
                                                      })
data["SimplBsmtCond"] = data.BsmtCond.replace({1 : 1, # bad
                                               2 : 1, 3 : 1, # average
                                               4 : 2, 5 : 2 # good
                                              })
data["SimplBsmtQual"] = data.BsmtQual.replace({1 : 1, # bad
                                               2 : 1, 3 : 1, # average
                                               4 : 2, 5 : 2 # good
                                              })
data["SimplExterCond"] = data.ExterCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })
data["SimplExterQual"] = data.ExterQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })




# 2* Combinations of existing features
# Overall quality of the house
data["OverallGrade"] = data["OverallQual"] * data["OverallCond"]
# Overall quality of the garage
data["GarageGrade"] = data["GarageQual"] * data["GarageCond"]
# Overall quality of the exterior
data["ExterGrade"] = data["ExterQual"] * data["ExterCond"]
# Overall kitchen score
data["KitchenScore"] = data["KitchenAbvGr"] * data["KitchenQual"]
# Overall fireplace score
data["FireplaceScore"] = data["Fireplaces"] * data["FireplaceQu"]
# Overall garage score
data["GarageScore"] = data["GarageArea"] * data["GarageQual"]
# Overall pool score
data["PoolScore"] = data["PoolArea"] * data["PoolQC"]
# Simplified overall quality of the house
data["SimplOverallGrade"] = data["SimplOverallQual"] * data["SimplOverallCond"]
# Simplified overall quality of the exterior
data["SimplExterGrade"] = data["SimplExterQual"] * data["SimplExterCond"]
# Simplified overall pool score
data["SimplPoolScore"] = data["PoolArea"] * data["SimplPoolQC"]
# Simplified overall garage score
data["SimplGarageScore"] = data["GarageArea"] * data["SimplGarageQual"]
# Simplified overall fireplace score
data["SimplFireplaceScore"] = data["Fireplaces"] * data["SimplFireplaceQu"]
# Simplified overall kitchen score
data["SimplKitchenScore"] = data["KitchenAbvGr"] * data["SimplKitchenQual"]
# Total number of bathrooms
data["TotalBath"] = data["BsmtFullBath"] + 0.5 * data["BsmtHalfBath"] + data["FullBath"] + 0.5 * data["HalfBath"]
# Total SF for house (incl. basement)
data["AllSF"] = data["GrLivArea"] + data["TotalBsmtSF"]
# Total SF for 1st + 2nd floors
data["AllFlrsSF"] = data["1stFlrSF"] + data["2ndFlrSF"]
# Total SF for porch
data["AllPorchSF"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]
# Has masonry veneer or not
data["HasMasVnr"] = data.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                             "Stone" : 1, "None" : 0})
# House completed before sale or not
data["BoughtOffPlan"] = data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1}) 
# House old or not 
data["Built"] = pd.Series(np.repeat("Old",data.shape[0]))
data.loc[data.YearBuilt > data.YearBuilt.mean(),"Built"] = "New"
# Month to categorical 
data["CatMoSold"] = data.MoSold.replace({1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                         7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"})
# Month number since 01/2006
data["Month"] = data.MoSold + 12*(data.YrSold - 2006) - 1




# creating matrices to use for correlations:
train2 = data[:train.shape[0]]
test2 = data[train.shape[0]:]
train2['SalePrice'] = train['SalePrice']

# Find most important features relative to target
print("Find most important features relative to target")
corr = train2.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)




# Create new features
# 3* Polynomials on the top 10 existing features
data["OverallQual-s2"] = data["OverallQual"] ** 2
data["OverallQual-s3"] = data["OverallQual"] ** 3
data["OverallQual-Sq"] = np.sqrt(data["OverallQual"])
data["AllSF-2"] = data["AllSF"] ** 2
data["AllSF-3"] = data["AllSF"] ** 3
data["AllSF-Sq"] = np.sqrt(data["AllSF"])
data["AllFlrsSF-2"] = data["AllFlrsSF"] ** 2
data["AllFlrsSF-3"] = data["AllFlrsSF"] ** 3
data["AllFlrsSF-Sq"] = np.sqrt(data["AllFlrsSF"])
data["GrLivArea-2"] = data["GrLivArea"] ** 2
data["GrLivArea-3"] = data["GrLivArea"] ** 3
data["GrLivArea-Sq"] = np.sqrt(data["GrLivArea"])
data["SimplOverallQual-s2"] = data["SimplOverallQual"] ** 2
data["SimplOverallQual-s3"] = data["SimplOverallQual"] ** 3
data["SimplOverallQual-Sq"] = np.sqrt(data["SimplOverallQual"])
data["ExterQual-2"] = data["ExterQual"] ** 2
data["ExterQual-3"] = data["ExterQual"] ** 3
data["ExterQual-Sq"] = np.sqrt(data["ExterQual"])
data["GarageCars-2"] = data["GarageCars"] ** 2
data["GarageCars-3"] = data["GarageCars"] ** 3
data["GarageCars-Sq"] = np.sqrt(data["GarageCars"])
data["TotalBath-2"] = data["TotalBath"] ** 2
data["TotalBath-3"] = data["TotalBath"] ** 3
data["TotalBath-Sq"] = np.sqrt(data["TotalBath"])
data["KitchenQual-2"] = data["KitchenQual"] ** 2
data["KitchenQual-3"] = data["KitchenQual"] ** 3
data["KitchenQual-Sq"] = np.sqrt(data["KitchenQual"])
data["GarageScore-2"] = data["GarageScore"] ** 2
data["GarageScore-3"] = data["GarageScore"] ** 3
data["GarageScore-Sq"] = np.sqrt(data["GarageScore"])




# Differentiate numerical features (minus the target) and categorical features
categorical_features = data.select_dtypes(include = ["object"]).columns
numerical_features = data.select_dtypes(exclude = ["object"]).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
data_num = data[numerical_features]
data_cat = data[categorical_features]




print("NAs for numerical features in train : " + str(data_num.isnull().values.sum()))
data_num = data_num.fillna(data_num.median())
print("Remaining NAs for numerical features in train : " + str(data_num.isnull().values.sum()))




# Alley : data description says NA means "no alley access"
data.loc[:, "Alley"] = data.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
data.loc[:, "BedroomAbvGr"] = data.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
data.loc[:, "BsmtQual"] = data.loc[:, "BsmtQual"].fillna("No")
data.loc[:, "BsmtCond"] = data.loc[:, "BsmtCond"].fillna("No")
data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna("No")
data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna("No")
data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna("No")
data.loc[:, "BsmtFullBath"] = data.loc[:, "BsmtFullBath"].fillna(0)
data.loc[:, "BsmtHalfBath"] = data.loc[:, "BsmtHalfBath"].fillna(0)
data.loc[:, "BsmtUnfSF"] = data.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
data.loc[:, "CentralAir"] = data.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
data.loc[:, "Condition1"] = data.loc[:, "Condition1"].fillna("Norm")
data.loc[:, "Condition2"] = data.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
data.loc[:, "EnclosedPorch"] = data.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
data.loc[:, "ExterCond"] = data.loc[:, "ExterCond"].fillna("TA")
data.loc[:, "ExterQual"] = data.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
data.loc[:, "Fence"] = data.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
data.loc[:, "FireplaceQu"] = data.loc[:, "FireplaceQu"].fillna("No")
data.loc[:, "Fireplaces"] = data.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
data.loc[:, "Functional"] = data.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna("No")
data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna("No")
data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna("No")
data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna("No")
data.loc[:, "GarageArea"] = data.loc[:, "GarageArea"].fillna(0)
data.loc[:, "GarageCars"] = data.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
data.loc[:, "HalfBath"] = data.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
data.loc[:, "HeatingQC"] = data.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
data.loc[:, "KitchenAbvGr"] = data.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
data.loc[:, "KitchenQual"] = data.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
data.loc[:, "LotShape"] = data.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
data.loc[:, "MasVnrType"] = data.loc[:, "MasVnrType"].fillna("None")
data.loc[:, "MasVnrArea"] = data.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
data.loc[:, "MiscFeature"] = data.loc[:, "MiscFeature"].fillna("No")
data.loc[:, "MiscVal"] = data.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
data.loc[:, "OpenPorchSF"] = data.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
data.loc[:, "PavedDrive"] = data.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
data.loc[:, "PoolQC"] = data.loc[:, "PoolQC"].fillna("No")
data.loc[:, "PoolArea"] = data.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
data.loc[:, "SaleCondition"] = data.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
data.loc[:, "ScreenPorch"] = data.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
data.loc[:, "TotRmsAbvGrd"] = data.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
data.loc[:, "Utilities"] = data.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
data.loc[:, "WoodDeckSF"] = data.loc[:, "WoodDeckSF"].fillna(0)




print("NAs for categorical features in dataset : " + str(data_cat.isnull().values.sum()))
data_cat = data[categorical_features]
print("Remaining NAs for categorical features in dataset : " + str(data_cat.isnull().values.sum()))




# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf
sns.regplot('GrLivArea','SalePrice',data=train2,ci=None,fit_reg=False)




data = data.loc[data.GrLivArea < 4000,:]




sns.distplot(y, kde = False, color = 'b',hist_kws={'alpha': 0.9})




sns.distplot(np.log(y+1), kde = False, color = 'b', hist_kws={'alpha': 0.9})




# Log transform the target for official scoring
y = np.log1p(train.SalePrice)




# Log transform of the skewed numerical features to lessen impact of outliers
# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
skewness = all_daply.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
data_num[skewed_features] = np.log1p(data_num[skewed_features])




# Standardize numerical features
stdSc = StandardScaler()
data.loc[:, numerical_features] = stdSc.fit_transform(data.loc[:, numerical_features])
data.loc[:, numerical_features] = stdSc.transform(data.loc[:, numerical_features])




# Create dummy features for categorical values via one-hot encoding
data_cat = pd.get_dummies(data_cat)




# Join categorical and numerical features
data = pd.concat([data_num, data_cat], axis = 1)
print("New number of features : " + str(data.shape[1]))

# Partition the dataset in train + validation sets
train = data[:train.shape[0]]
test = data[train.shape[0]:]
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))




X_train = X_train.iloc[:,optimal_indexes]
X_test = X_test.iloc[:,optimal_indexes]
xtrain, xtest, ytrain, ytest = train_test_split(X_train, y, test_size=0.2, random_state=43)




regr.fit(xtrain, np.log(ytrain+1))
print (cross_val_score(regr, X_train, y, cv=10).mean())
print(np.mean((regr.predict(xtest) - np.log(ytest+1)) ** 2))




ridge = Ridge(max_iter=5000, alpha=0.4, normalize=True)
ridge.fit(xtrain.loc[:,['GrLivAre','TotalBsmtSF']], np.log(ytrain+1))
print (cross_val_score(ridge, X_train, y, cv=10).mean())
print(np.mean((ridge.predict(xtest) - np.log(ytest+1)) ** 2))




# Regression with ridge regularization (L2 penalty)
parameters = {'alpha': [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60], # 0.3 / True
              'normalize': (True, False)}
# create and fit a ridge regression model, testing random alpha values
ridge = GridSearchCV(Ridge(), parameters)
ridge.fit(xtrain, np.log(ytrain+1))
# summarize the results of the random parameter search
print(ridge.best_score_)
print(mean_squared_error(np.log(ytest+1),ridge.predict(xtest)))
print(ridge.best_estimator_)




# Regression with lasso regularization (L1 penalty)
parameters = {'alpha': [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,0.3, 0.6, 1], # 0.5/ 'sag' / T
              'normalize': (True, False)} # true / 0.0001
# create and fit a  regression model, testing random alpha values
lasso = GridSearchCV(Lasso(max_iter=5000), parameters)
lasso.fit(xtrain, np.log(ytrain+1))
# summarize the results of the random parameter search
print(lasso.best_score_)
print(mean_squared_error(np.log(ytest+1),lasso.predict(xtest)))
print(lasso.best_estimator_)




# regression with elasticnet regularization (L1 and L2 penalty)
parameters = {'alpha': [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
              'l1_ratio' : [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
              'normalize': (True, False)} # true / 0.0003 / ratio= 0.1
# create and fit a  regression model, testing random alpha values
elastic = GridSearchCV(ElasticNet(max_iter=50000), parameters)
elastic.fit(xtrain, np.log(ytrain+1))
# summarize the results of the random parameter search
print(elastic.best_score_)
print(mean_squared_error(np.log(ytest+1),elastic.predict(xtest)))
print(elastic.best_estimator_)




# Neural Network
parameters = {'hidden_layer_sizes': [(100,100),(60,60),(50,50)], #(64,64) / "ranh"
              'activation': ['identity', 'logistic', 'tanh', 'relu']}
# create and fit a  model, testing random alpha values
mlp = GridSearchCV(MLPRegressor(), parameters)
mlp.fit(xtrain, np.log(ytrain+1))
# summarize the results of the random parameter search
print(mlp.best_score_)
print(mean_squared_error(np.log(ytest+1),mlp.predict(xtest)))
print(mlp.best_estimator_)




# Random forest
parameters = {'n_estimators': np.arange(50,250,50), # 150, 30
              'max_depth':np.arange(20,40,5)}
# create and fit a model, testing random alpha values
rf = GridSearchCV(RandomForestRegressor(), parameters)
rf.fit(xtrain, np.log(ytrain+1))
# summarize the results of the random parameter search
print(rf.best_score_)
print(mean_squared_error(np.log(ytest+1),rf.predict(xtest)))
print(rf.best_estimator_)




# Gradient Boosting Regressor
parameters = {'n_estimators': np.arange(50,250,50), # 150, 30, 1
              'max_depth':np.arange(20,40,5),
             'learning_rate': np.arange(0.01,1,0.01)}

# create and fit a  model, testing random alpha values
gbr = GridSearchCV(GradientBoostingRegressor(), parameters)
gbr.fit(xtrain, np.log(ytrain+1))
# summarize the results of the random parameter search
print(gbr.best_score_)
print(mean_squared_error(np.log(ytest+1),gbr.predict(xtest)))
print(gbr.best_estimator_)




model_xgb = xgb.XGBRegressor(n_estimators=150, max_depth=20, learning_rate=0.1) 
model_xgb.fit(xtrain, np.log(ytrain+1))

#print(model_xgb.best_score_)
print(mean_squared_error(np.log(ytest+1),model_xgb.predict(xtest)))
#print(model_xgb.best_estimator_)




# create the sub models (STACKING) bagging?
estimators = []
ridge = Ridge(alpha=0.5,solver= 'sag', normalize= True)
estimators.append(('ridge',ridge))
mlp = MLPRegressor()
estimators.append(('mlp', mlp))
rf = RandomForestRegressor(n_estimators=150,max_depth=30)
estimators.append(('rf', rf))
gbr = GrandiantBoostingRegressor()
estimators.append(('gbr', gbr))
# create the ensemble model
ensemble = VotingClassifier(estimators)
kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)
results = cross_val_score(ensemble, xtrain, ytrain, cv=kfold)
print(results.mean())




train_sizes, train_scores, test_scores = learning_curve(regr, X_train, np.log(y+1), 
                                                        train_sizes=[1, 600, 1000], cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")
plt.legend(loc="best")




rf = RandomForestRegressor(n_estimators=150,max_depth=30)
rf.fit(xtrain, np.log(ytrain+1))
ridge= Ridge(alpha=0.5,solver= 'sag', normalize= True)
ridge.fit(xtrain, np.log(ytrain+1))




# save to file to make a submission
p = np.expm1(ridge.predict(X_test))
solution = pd.DataFrame({"id":test.Id, "SalePrice":p*1000}, columns=['id', 'SalePrice'])
solution.to_csv("regression_sol.csv", index = False)

