#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)
from scipy import stats
import warnings
warnings.filterwarnings('ignore') # hiding warning messages for readability
# warnings.filterwarnings("ignore", category=DeprecationWarning)




train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')




train.head()




test.head()




train.shape, test.shape




train.columns




train.get_dtype_counts()




train.describe()




train.info()




corr = train.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]




plt.figure(figsize=(20,20))
corr = corr[1:-1] # removing 1st (SalePrice) and last (Id) row from dataframe
corr.plot(kind='barh') # using pandas plot
plt.title('Correlation coefficients w.r.t. Sale Price')




# taking high correlated variables having positive correlation of 45% and above
high_positive_correlated_variables = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',                                'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',                                'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces']

corrMatrix = train[high_positive_correlated_variables].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(15, 15))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True, annot=True, cmap='viridis', linecolor="white")

plt.title('Correlation between features');




feature_variable = 'OverallQual'
target_variable = 'SalePrice'
train[[feature_variable, target_variable]].groupby([feature_variable], as_index=False).mean().sort_values(by=feature_variable, ascending=False)




feature_variable = 'GarageCars'
target_variable = 'SalePrice'
train[[feature_variable, target_variable]].groupby([feature_variable], as_index=False).mean().sort_values(by=feature_variable, ascending=False)




cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)




# box plot overallqual/saleprice
plt.figure(figsize=[10,5])
sns.boxplot(x='OverallQual', y="SalePrice", data=train)




train['SalePrice'].describe()




# histogram to graphically show skewness and kurtosis
plt.figure(figsize=[15,5])
sns.distplot(train['SalePrice'])
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Number of Occurences')




# normal probability plot
plt.figure(figsize=[8,6])
stats.probplot(train['SalePrice'], plot=plt)




# skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())




plt.figure(figsize=[8,6])
plt.scatter(train["SalePrice"].values, range(train.shape[0]))
plt.title("Distribution of Sale Price")
plt.xlabel("Sale Price");
plt.ylabel("Number of Occurences")




# removing outliers
upperlimit = np.percentile(train.SalePrice.values, 99.5)
train['SalePrice'].loc[train['SalePrice']>upperlimit] = upperlimit # slicing dataframe upto the uppperlimit




# plotting again the graph after removing outliers
plt.figure(figsize=[8,6])
plt.scatter(train["SalePrice"].values, range(train.shape[0]))
plt.title("Distribution of Sale Price")
plt.xlabel("Sale Price");
plt.ylabel("Number of Occurences")




# applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])




# histogram to graphically show skewness and kurtosis
plt.figure(figsize=[15,5])
sns.distplot(train['SalePrice'])
plt.title('Distribution of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Number of Occurences')

# normal probability plot
plt.figure(figsize=[8,6])
stats.probplot(train['SalePrice'], plot=plt)




# skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())




sns.factorplot(x="PoolArea",y="SalePrice",data=train,hue="PoolQC",kind='bar')
plt.title("Pool Area , Pool quality and SalePrice ")
plt.ylabel("SalePrice")
plt.xlabel("Pool Area in sq feet");




sns.factorplot("Fireplaces","SalePrice",data=train,hue="FireplaceQu");




pd.crosstab(train.Fireplaces, train.FireplaceQu)




# scatter plot grlivarea/saleprice
plt.figure(figsize=[8,6])
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)




# Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)




# Plot the graph again

# scatter plot grlivarea/saleprice
plt.figure(figsize=[8,6])
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.xlabel('GrLivArea', fontsize=13)
plt.ylabel('SalePrice', fontsize=13)




ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.shape




null_columns = all_data.columns[all_data.isnull().any()]
total_null_columns = all_data[null_columns].isnull().sum()
percent_null_columns = ( all_data[null_columns].isnull().sum() / all_data[null_columns].isnull().count() )
missing_data = pd.concat([total_null_columns, percent_null_columns], axis=1, keys=['Total', 'Percent']).sort_values(by=['Percent'], ascending=False)
#missing_data.head()
missing_data




plt.figure(figsize=[20,5])
plt.xticks(rotation='90', fontsize=14)
sns.barplot(x=missing_data.index, y=missing_data.Percent)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)




# get unique values of the column data
all_data['PoolQC'].unique()




# replace null values with 'None'
all_data['PoolQC'].fillna('None', inplace=True)




# get unique values of the column data
all_data['PoolQC'].unique()




# get unique values of the column data
all_data['MiscFeature'].unique()




# replace null values with 'None'
all_data['MiscFeature'].fillna('None', inplace=True)




# get unique values of the column data
all_data['Alley'].unique()




# replace null values with 'None'
all_data['Alley'].fillna('None', inplace=True)




# get unique values of the column data
all_data['Fence'].unique()




# replace null values with 'None'
all_data['Fence'].fillna('None', inplace=True)




# get unique values of the column data
all_data['FireplaceQu'].unique()




# replace null values with 'None'
all_data['FireplaceQu'].fillna('None', inplace=True)




# barplot of median of LotFrontage with respect to Neighborhood
sns.barplot(data=train,x='Neighborhood',y='LotFrontage', estimator=np.median)
plt.xticks(rotation=90)




# get unique values of the column data
all_data['LotFrontage'].unique()




# replace null values with median LotFrontage of all the Neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))




all_data['LotFrontage'].unique()




# get unique values of the column data
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    print (all_data[col].unique())




# replace null values with 'None'
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col].fillna('None', inplace=True)




# get unique values of the column data
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    print (all_data[col].unique())




# replace null values with 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col].fillna(0, inplace=True)




# get unique values of the column data
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    print (all_data[col].unique())




# replace null values with 'None'
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col].fillna('None', inplace=True)




# replace null values with 0
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col].fillna(0, inplace=True)




all_data["MasVnrType"].fillna("None", inplace=True)
all_data["MasVnrArea"].fillna(0, inplace=True)




for col in ('MSZoning', 'Utilities', 'Functional', 'Exterior2nd', 'Exterior1st', 'KitchenQual', 'Electrical', 'SaleType'):
    all_data[col].fillna(all_data[col].mode()[0], inplace=True)




null_columns = all_data.columns[all_data.isnull().any()]
print (null_columns)




numeric_features = all_data.dtypes[all_data.dtypes != 'object'].index
#print (numeric_features)

skewness = []
for col in numeric_features:
    skewness.append( (col, all_data[col].skew()) )
    
pd.DataFrame(skewness, columns=('Feature', 'Skewness')).sort_values(by='Skewness', ascending=False)




all_data.head()




positively_skewed_features = all_data[numeric_features].columns[abs(all_data[numeric_features].skew()) > 1]
#print (positively_skewed_features)

# applying log transformation
for col in positively_skewed_features:
    all_data[col] = np.log(np.ma.array(all_data[col], mask=(all_data[col]<=0))) # using masked array to ignore log transformation of 0 values as (log 0) is undefined




all_data.head()




get_ipython().run_cell_magic('HTML', '', '<style>\n  table {margin-left: 0 !important;}\n</style>')




all_data = pd.get_dummies(all_data)
print(all_data.shape)




train = all_data[:ntrain]
test = all_data[ntrain:]




train.head()




test.head()




# importing model libraries
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb




X_train = train.drop(['Id'], axis=1)
# y_train has been defined above where we combined train and test data to create all_data
X_test = test.drop(['Id'], axis=1)




#lasso = Lasso(alpha =0.0005, random_state=1)
#lasso = Lasso()
model_lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005))
# y_train is defined above where we combined train and test data to create all_data
# np.sqrt() function is used to create square root of MSE returned by cross_val_score function
cv_score = np.sqrt( -cross_val_score(model_lasso, X_train, y_train, scoring="neg_mean_squared_error", cv=5) )
print (cv_score)
print ("SCORE (mean: %f , std: %f)" % (np.mean(cv_score), np.std(cv_score)))




model_elastic_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005))
# y_train is defined above where we combined train and test data to create all_data
# np.sqrt() function is used to create square root of MSE returned by cross_val_score function
cv_score = np.sqrt( -cross_val_score(model_elastic_net, X_train, y_train, scoring="neg_mean_squared_error", cv=5) )
print (cv_score)
print ("SCORE (mean: %f , std: %f)" % (np.mean(cv_score), np.std(cv_score)))




model_kernel_ridge = KernelRidge(alpha=0.6)
# y_train is defined above where we combined train and test data to create all_data
# np.sqrt() function is used to create square root of MSE returned by cross_val_score function
cv_score = np.sqrt( -cross_val_score(model_kernel_ridge, X_train, y_train, scoring="neg_mean_squared_error", cv=5) )
print (cv_score)
print ("SCORE (mean: %f , std: %f)" % (np.mean(cv_score), np.std(cv_score)))




model_gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state=5)

# y_train is defined above where we combined train and test data to create all_data
# np.sqrt() function is used to create square root of MSE returned by cross_val_score function
cv_score = np.sqrt( -cross_val_score(model_gboost, X_train, y_train, scoring="neg_mean_squared_error", cv=5) )
print (cv_score)
print ("SCORE (mean: %f , std: %f)" % (np.mean(cv_score), np.std(cv_score)))




model_xgboost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=True, nthread = -1)

# y_train is defined above where we combined train and test data to create all_data
# np.sqrt() function is used to create square root of MSE returned by cross_val_score function
cv_score = np.sqrt( -cross_val_score(model_xgboost, X_train, y_train, scoring="neg_mean_squared_error", cv=5) )
print (cv_score)
print ("SCORE (mean: %f , std: %f)" % (np.mean(cv_score), np.std(cv_score)))




model_lgbm = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# y_train is defined above where we combined train and test data to create all_data
# np.sqrt() function is used to create square root of MSE returned by cross_val_score function
cv_score = np.sqrt( -cross_val_score(model_lgbm, X_train, y_train, scoring="neg_mean_squared_error", cv=5) )
print (cv_score)
print ("SCORE (mean: %f , std: %f)" % (np.mean(cv_score), np.std(cv_score)))




model_lasso.fit(X_train, y_train)
model_elastic_net.fit(X_train, y_train)
model_kernel_ridge.fit(X_train, y_train)
model_gboost.fit(X_train, y_train)
model_xgboost.fit(X_train, y_train)
model_lgbm.fit(X_train, y_train)




dict_models = {'lasso':model_lasso, 'elastic_net':model_elastic_net, 
               'kernel_ridge':model_kernel_ridge, 
               'gboost':model_gboost, 'xgboost':model_xgboost, 'lgbm':model_lgbm}

for key, value in dict_models.items():
    pred_train = value.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    print ("%s: %f" % (key, rmse))




prediction_lasso = np.expm1(model_lasso.predict(X_test))
prediction_elastic_net = np.expm1(model_elastic_net.predict(X_test))
prediction_kernel_ridge = np.expm1(model_kernel_ridge.predict(X_test))
prediction_gboost = np.expm1(model_gboost.predict(X_test))
prediction_xgboost = np.expm1(model_xgboost.predict(X_test))
prediction_lgbm = np.expm1(model_lgbm.predict(X_test))




# kaggle score: 0.12346
#prediction = prediction_gboost

# kaggle score: 0.12053
#prediction = (prediction_lasso + prediction_xgboost) / float(2) 

# kaggle score: 0.11960
#prediction = prediction_lasso 

# kaggle score: 0.11937
prediction = (prediction_lasso + prediction_elastic_net) / float(2) 

#print prediction




submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": prediction
    })

#submission.to_csv('submission.csv', index=False)

