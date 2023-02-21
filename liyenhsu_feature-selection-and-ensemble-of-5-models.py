#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import boxcox
from scipy.stats import skew
from scipy.stats import randint
from scipy.stats import uniform

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer 
from sklearn.base import BaseEstimator, RegressorMixin

# neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

# ignore Deprecation Warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# load the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df = df_train.append(df_test , ignore_index = True)

# basic inspection
df_train.shape, df_test.shape, df_train.columns.values




# divide the data into numerical ("quan") and categorical ("qual") features
quan = list( df_test.loc[:,df_test.dtypes != 'object'].drop('Id',axis=1).columns.values )
qual = list( df_test.loc[:,df_test.dtypes == 'object'].columns.values )




# Find out the missing values for quantitative and categorical features

hasNAN = df[quan].isnull().sum()
hasNAN = hasNAN[hasNAN > 0]
hasNAN = hasNAN.sort_values(ascending=False)
print(hasNAN)

print('**'*40)

hasNAN = df[qual].isnull().sum()
hasNAN = hasNAN[hasNAN > 0]
hasNAN = hasNAN.sort_values(ascending=False)
print(hasNAN)




df.SalePrice = np.log(df.SalePrice)




# Filling missing values for numerical features
# Most of the NAN should mean that the corresponding facillity/structure doesn't 
# exist, so I use zero for most cases
df.LotFrontage.fillna(df.LotFrontage.median(), inplace=True)

# NAN should mean no garage. I temporarily use yr =0 here. Will come back to this later. 
df.GarageYrBlt.fillna(0, inplace=True)

df.MasVnrArea.fillna(0, inplace=True)    
df.BsmtHalfBath.fillna(0, inplace=True)
df.BsmtFullBath.fillna(0, inplace=True)
df.GarageArea.fillna(0, inplace=True)
df.GarageCars.fillna(0, inplace=True)    
df.TotalBsmtSF.fillna(0, inplace=True)   
df.BsmtUnfSF.fillna(0, inplace=True)     
df.BsmtFinSF2.fillna(0, inplace=True)    
df.BsmtFinSF1.fillna(0, inplace=True)    

# categorical features
df.PoolQC.fillna('NA', inplace=True)
df.MiscFeature.fillna('NA', inplace=True)    
df.Alley.fillna('NA', inplace=True)          
df.Fence.fillna('NA', inplace=True)         
df.FireplaceQu.fillna('NA', inplace=True)    
df.GarageCond.fillna('NA', inplace=True)    
df.GarageQual.fillna('NA', inplace=True)     
df.GarageFinish.fillna('NA', inplace=True)   
df.GarageType.fillna('NA', inplace=True)     
df.BsmtExposure.fillna('NA', inplace=True)     
df.BsmtCond.fillna('NA', inplace=True)        
df.BsmtQual.fillna('NA', inplace=True)        
df.BsmtFinType2.fillna('NA', inplace=True)     
df.BsmtFinType1.fillna('NA', inplace=True)     
df.MasVnrType.fillna('None', inplace=True)   
df.Exterior2nd.fillna('None', inplace=True) 

# These are general properties that all houses should have, so NAN probably 
# just means the value was not recorded. I therefore use "mode", the most 
# common value to fill in
df.Functional.fillna(df.Functional.mode()[0], inplace=True)       
df.Utilities.fillna(df.Utilities.mode()[0], inplace=True)          
df.Exterior1st.fillna(df.Exterior1st.mode()[0], inplace=True)        
df.SaleType.fillna(df.SaleType.mode()[0], inplace=True)                
df.KitchenQual.fillna(df.KitchenQual.mode()[0], inplace=True)        
df.Electrical.fillna(df.Electrical.mode()[0], inplace=True)    

# MSZoning should highly correlate with the location, so I use the mode values of individual 
# Neighborhoods
for i in df.Neighborhood.unique():
    if df.MSZoning[df.Neighborhood == i].isnull().sum() > 0:
        df.loc[df.Neighborhood == i,'MSZoning'] =         df.loc[df.Neighborhood == i,'MSZoning'].fillna(df.loc[df.Neighborhood == i,'MSZoning'].mode()[0]) 

# These categorical features are "rank", so they can be transformed to 
# numerical features
df.Alley = df.Alley.map({'NA':0, 'Grvl':1, 'Pave':2})
df.BsmtCond =  df.BsmtCond.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.BsmtExposure = df.BsmtExposure.map({'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})
df['BsmtFinType1'] = df['BsmtFinType1'].map({'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
df['BsmtFinType2'] = df['BsmtFinType2'].map({'NA':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
df.BsmtQual = df.BsmtQual.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.ExterCond = df.ExterCond.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.ExterQual = df.ExterQual.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.FireplaceQu = df.FireplaceQu.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.Functional = df.Functional.map({'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8})
df.GarageCond = df.GarageCond.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.GarageQual = df.GarageQual.map({'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.HeatingQC = df.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.KitchenQual = df.KitchenQual.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
df.LandSlope = df.LandSlope.map({'Sev':1, 'Mod':2, 'Gtl':3}) 
df.PavedDrive = df.PavedDrive.map({'N':1, 'P':2, 'Y':3})
df.PoolQC = df.PoolQC.map({'NA':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
df.Street = df.Street.map({'Grvl':1, 'Pave':2})
df.Utilities = df.Utilities.map({'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4})

# Update my lists of numerical and categorical features
newquan = ['Alley','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual',
           'ExterCond','ExterQual','FireplaceQu','Functional','GarageCond',
           'GarageQual','HeatingQC','KitchenQual','LandSlope','PavedDrive','PoolQC',
           'Street','Utilities']
quan = quan + newquan 
for i in newquan: qual.remove(i)


# This is actually a categorical feature...
df.MSSubClass = df.MSSubClass.map({20:'class1', 30:'class2', 40:'class3', 45:'class4',
                                   50:'class5', 60:'class6', 70:'class7', 75:'class8',
                                   80:'class9', 85:'class10', 90:'class11', 120:'class12',
                                   150:'class13', 160:'class14', 180:'class15', 190:'class16'})

# Keeping "YrSold" is enough.
df=df.drop('MoSold',axis=1)

# Update my lists of numerical and categorical features
quan.remove('MoSold')
quan.remove('MSSubClass')
qual.append('MSSubClass')




df['Age'] = df.YrSold - df.YearBuilt
df['AgeRemod'] = df.YrSold - df.YearRemodAdd
df['AgeGarage'] = df.YrSold - df.GarageYrBlt

# For the houses without a Garage, I filled the NANs with zeros, which makes AgeGarage ~ 2000
# Here I replace their AgeGarage with the maximum value among the houses with Garages
max_AgeGarage = np.max(df.AgeGarage[df.AgeGarage < 1000])
df['AgeGarage'] = df['AgeGarage'].map(lambda x: max_AgeGarage if x > 1000 else x)

# Some of the values are negative because the work was done after the house 
# was sold. In these cases, I change them to zero to avoid negative ages.
df.Age = df.Age.map(lambda x: 0 if x < 0 else x)
df.AgeRemod = df.AgeRemod.map(lambda x: 0 if x < 0 else x)
df.AgeGarage = df.AgeGarage.map(lambda x: 0 if x < 0 else x)

# drop the original time variables 
df=df.drop(['YrSold','YearBuilt','YearRemodAdd','GarageYrBlt'],axis=1)

# update my list of numerical feature
quan.remove('YrSold')
quan.remove('YearBuilt')
quan.remove('YearRemodAdd')
quan.remove('GarageYrBlt')
quan = quan + ['Age','AgeRemod','AgeGarage']




# visualize the distribution of each numerical feature
temp = pd.melt(df.drop('SalePrice',axis=1), value_vars=quan)
grid = sns.FacetGrid(temp, col="variable",  col_wrap=6 , size=3.0, 
                     aspect=0.8,sharex=False, sharey=False)
grid.map(sns.distplot, "value")
plt.show()

# scatter plots
temp = pd.melt(df, id_vars=['SalePrice'],value_vars=quan)
grid = sns.FacetGrid(temp, col="variable",  col_wrap=4 , size=3.0, 
                     aspect=1.2,sharex=False, sharey=False)
grid.map(plt.scatter, "value",'SalePrice',s=1.5)
plt.show()




# print the skewness of each numerical feature
for i in quan:
    print(i+':', round(skew(df[i]),2) ) 




# transform those with skewness > 0.5
skewed_features = np.array(quan)[np.abs(skew(df[quan])) > 0.5]
df[skewed_features] = np.log1p(df[skewed_features])




## visualize the distribution again
temp = pd.melt(df, value_vars=quan)
grid = sns.FacetGrid(temp, col="variable",  col_wrap=6 , size=3.0, 
                     aspect=0.8,sharex=False, sharey=False)
grid.map(sns.distplot, "value")
plt.show()

# scatter plots
temp = pd.melt(df, id_vars=['SalePrice'],value_vars=quan)
grid = sns.FacetGrid(temp, col="variable",  col_wrap=4 , size=3.0, 
                     aspect=1.2,sharex=False, sharey=False)
grid.map(plt.scatter, "value",'SalePrice',s=1.5)
plt.show()




# create of list of dummy variables that I will drop, which will be the last
# column generated from each categorical feature
dummy_drop = []
for i in qual:
    dummy_drop += [ i+'_'+str(df[i].unique()[-1]) ]

# create dummy variables
df = pd.get_dummies(df,columns=qual) 
# drop the last column generated from each categorical feature
df = df.drop(dummy_drop,axis=1)




X_train  = df[:1460].drop(['SalePrice','Id'], axis=1)
y_train  = df[:1460]['SalePrice']
X_test  = df[1460:].drop(['SalePrice','Id'], axis=1)

# fit the training set only, then transform both the training and test sets
scaler = RobustScaler()
X_train[quan]= scaler.fit_transform(X_train[quan])
X_test[quan]= scaler.transform(X_test[quan])

X_train.shape # now we have 221 features!




xgb = XGBRegressor()
xgb.fit(X_train, y_train)
imp = pd.DataFrame(xgb.feature_importances_ ,columns = ['Importance'],index = X_train.columns)
imp = imp.sort_values(['Importance'], ascending = False)

print(imp)




# Define a function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

# Define a function to calculate negative RMSE (as a score)
def nrmse(y_true, y_pred):
    return -1.0*rmse(y_true, y_pred)

neg_rmse = make_scorer(nrmse)

estimator = XGBRegressor()
selector = RFECV(estimator, cv = 3, n_jobs = -1, scoring = neg_rmse)
selector = selector.fit(X_train, y_train)

print("The number of selected features is: {}".format(selector.n_features_))

features_kept = X_train.columns.values[selector.support_] 

X_train = selector.transform(X_train)  
X_test = selector.transform(X_test)

# transform it to a numpy array so later we can feed it to a neural network
y_train = y_train.values 




"""
xgb = XGBRegressor()
scoring_fnc = make_scorer(rmse)

best_rmse = 100 # initialize the best rmse, which will be updated in the loop
n_feat = 0  # initialize the number of features we choose, which will be updated in the loop

rmse_cv = [] # for recording RMSE

# shuffle the data first
from sklearn.utils import shuffle
X_shuffled, y_shuffled = shuffle(X_train, y_train, random_state=0) 

# start from the top 10 features, then add more less important ones
for i in range(10,len(imp)+1):

    keep = imp.iloc[0:i].index.values
    X_temp = X_shuffled[keep] 

    scores = cross_val_score(xgb, X_temp, y_shuffled, scoring=scoring_fnc)
    rms = scores.mean() # mean rmse of the three k-folds
    rmse_cv += [rms]

    # include more features only if RMES improves by more than 0.01% 
    if (rms - best_rmse)/best_rmse < -1e-4:
        best_rmse = rms
        n_feat = len(keep)
        feat = keep

# plot RMES v.s. number of features
fig = plt.figure()
plt.plot(range(n_min,len(imp)+1),rmse_cv)
plt.xlabel('# of Features')
plt.ylabel('RMSE')
plt.show(block=False)
        
# final number of features we will use = 87
print(n_feat) 
X_train = X_train[feat]        
X_test = X_test[feat]     

"""




# XGBoost: LB score = 0.12431
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, subsample=0.5, colsample_bytree=0.5, 
                   max_depth=3, gamma=0, reg_alpha=0, reg_lambda=2, min_child_weight=1)

# Lasso: LB score = 0.12568
las = Lasso(alpha=0.00049, max_iter=50000) 

# Elastic Net: LB score = 0.12651
elast = ElasticNet(alpha=0.0003, max_iter=50000, l1_ratio=0.83) 

# Kernel Ridge: LB score = 0.12570
ridge = KernelRidge(alpha=0.15, coef0=3.7, degree=2, kernel='polynomial')

# Gradient Boosting: LB score > 0.13 --> decided not to use it
# boost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.042, subsample=0.5, 
#       random_state=0, min_samples_split=4, max_depth=4)

# Neural Network: LB score = 0.12064
nn = Sequential()

# layers
nn.add(Dense(units = 40, kernel_initializer = 'uniform', activation = 'relu',
             input_dim = X_train.shape[1], kernel_regularizer=regularizers.l2(0.003)))
nn.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu',
             kernel_regularizer=regularizers.l2(0.003)))
nn.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu',
             kernel_regularizer=regularizers.l2(0.003)))
nn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu',
             kernel_regularizer=regularizers.l2(0.003)))

# Compile the NN
nn.compile(loss='mean_squared_error', optimizer='sgd')




class Ensemble(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors
        
    def level0_to_level1(self, X):
        self.predictions_ = []

        for regressor in self.regressors:
            self.predictions_.append(regressor.predict(X).reshape(X.shape[0],1))

        return np.concatenate(self.predictions_, axis=1)
    
    def fit(self, X, y):
        for regressor in self.regressors:
            if regressor != nn:
                regressor.fit(X, y)
            else: regressor.fit(X, y, batch_size=64, epochs=1000, verbose=0) # Neural Network
            
        self.new_features = self.level0_to_level1(X)
        
        # using a large L2 regularization to prevent the ensemble from biasing toward 
        # one particular base model
        self.combine = Ridge(alpha=10, max_iter=50000)   
        self.combine.fit(self.new_features, y)

        self.coef_ = self.combine.coef_

    def predict(self, X):
        self.new_features = self.level0_to_level1(X)
            
        return self.combine.predict(self.new_features).reshape(X.shape[0])




model = Ensemble(regressors=[xgb, las, elast, ridge, nn])
model.fit(X_train, y_train)
y_pred = np.exp(model.predict(X_test))

print("\nThe weights of the five base models are: {}".format(model.coef_))

output = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': y_pred})
output.to_csv('prediction-ensemble.csv', index=False)

