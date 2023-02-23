#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('HTML', '', "<h1>Majdoubi's Guide to a beginner's first Kaggle Project (top 20%)</h1>\n<h3>Made by : Ahmed Amine MAJDOUBI</h3>")




get_ipython().run_cell_magic('HTML', '', '<h1>Importing Libraries and Datasets</h1>')




#importing librairies

import pandas as pd # data processing
import numpy as np # numeric computation
import matplotlib.pyplot as plt # plot visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # data visualisation
color = sns.color_palette()
sns.set_style('darkgrid')
import scipy.stats as st # statistics
pd.options.display.max_columns = None # show all columns
import missingno as msno # missing data visualizations and utilities
import warnings # ignore file warnings
warnings.filterwarnings('ignore')

# Importing the train and test datasets in  pandas dataframe

trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')
trainData.drop(columns = 'Id', inplace =True)
y_train = trainData['SalePrice']




get_ipython().run_cell_magic('HTML', '', '<h1>Data Description</h1>')




# shape of data
trainData.shape , testData.shape, y_train.shape




# A basic description of the dataset
trainData.describe()




# display the first five rows of the train dataset.
trainData.head()




# display the last five rows of the train dataset.
trainData.tail()




# Numeric and categorical features in the dataset
trainData.select_dtypes(include=[np.number]).columns, trainData.select_dtypes(include=[np.object]).columns




get_ipython().run_cell_magic('HTML', '', '<h1>Correlation Features</h1>')




# Showing the numerical varibales with the highest correlation with 'SalePrice', sorted from highest to lowest
correlation = trainData.select_dtypes(include=[np.number]).corr()
print(correlation['SalePrice'].sort_values(ascending = False))




# Heatmap of correlation of numeric features
f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of numeric features',size=15)
sns.heatmap(correlation,square = True,  vmax=0.8)




# Zoomed HeatMap of the most Correlayed variables
zoomedCorrelation = correlation.loc[['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','FullBath','TotRmsAbvGrd','YearBuilt','1stFlrSF','GarageYrBlt','GarageCars','GarageArea'], ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','FullBath','TotRmsAbvGrd','YearBuilt','1stFlrSF','GarageYrBlt','GarageCars','GarageArea']]
f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of numeric features',size=15)
sns.heatmap(zoomedCorrelation, square = True, linewidths=0.01, vmax=0.8, annot=True,cmap='viridis',
            linecolor="black", annot_kws = {'size':12})




# Pair plot
sns.set()
cols = ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual','FullBath','TotRmsAbvGrd','YearBuilt','1stFlrSF','GarageYrBlt','GarageCars','GarageArea']
sns.pairplot(trainData[cols],size = 2 ,kind ='scatter',diag_kind='kde')
plt.show()




get_ipython().run_cell_magic('HTML', '', '<h1>Removing Outliers</h1>')




plt.figure(figsize=(7,5))
plt.scatter(x = trainData.TotalBsmtSF,y = trainData.SalePrice)
plt.title('TotalBsmtSF', size = 15)
plt.figure(figsize=(7,5))
plt.scatter(x = trainData['1stFlrSF'],y = trainData.SalePrice)
plt.title('1stFlrSF', size = 15)
plt.figure(figsize=(7,5))
plt.scatter(x = trainData.GrLivArea,y = trainData.SalePrice)
plt.title('GrLivArea', size = 15)




# Removing the outliers
trainData.drop(trainData[trainData['TotalBsmtSF'] > 5000].index,inplace = True)
trainData.drop(trainData[trainData['1stFlrSF'] > 4000].index,inplace = True)
trainData.drop(trainData[(trainData['GrLivArea'] > 4000) & (trainData['SalePrice']<300000)].index,inplace = True)
trainData.shape




get_ipython().run_cell_magic('HTML', '', '<h1>Imputation of missing values</h1>')




# Visualising missing values of numeric features for sample of 200
msno.matrix(trainData.select_dtypes(include=[np.number]).sample(200))




# Visualising percentage of missing values of the top 10 numeric variables
total = trainData.select_dtypes(include=[np.number]).isnull().sum().sort_values(ascending=False)
percent = (trainData.select_dtypes(include=[np.number]).isnull().sum()/trainData.select_dtypes(include=[np.number]).isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Missing Count', 'Missing Percentage'])
missing_data.index.name =' Numeric Feature'
missing_data.head(10)




# Visualising missing values of categorical features for sample of 200
msno.matrix(trainData.select_dtypes(include=[np.object]).sample(200))




# Visualising percentage of missing values of the top 10 categorical variables
total = trainData.select_dtypes(include=[np.object]).isnull().sum().sort_values(ascending=False)
percent = (trainData.select_dtypes(include=[np.object]).isnull().sum()/trainData.select_dtypes(include=[np.object]).isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1,join='outer', keys=['Missing Count', 'Missing Percentage'])
missing_data.index.name =' Numeric Feature'
missing_data.head(10)




# Visualization of nullity by column
msno.bar(trainData.sample(1000))




# Nullity correlation heatmap : how strongly the presence or absence of one variable affects the presence of another
msno.heatmap(trainData)

# -1 : if one variable appears the other definitely does not
# 0 : variables appearing or not appearing have no effect on one another 
# 1 : if one variable appears the other definitely also does




# Dendogram for variable completion, reveals trends deeper than the pairwise ones visible in the correlation heatmap
msno.dendrogram(trainData)




# Concatenate the training and test datasets into a single datafram
dataFull = pd.concat([trainData,testData],ignore_index=True)
dataFull.drop('Id',axis = 1,inplace = True)
dataFull.shape




# Sum of missing values by feature
sumMissingValues = dataFull.isnull().sum()
sumMissingValues[sumMissingValues>0].sort_values(ascending = False)




# Numeric features : replace with 0
for col in ['BsmtFullBath','BsmtHalfBath','BsmtUnfSF','TotalBsmtSF','GarageCars','BsmtFinSF2','BsmtFinSF1','GarageArea']:
    dataFull[col].fillna(0,inplace= True)

# Check if missing values are imputed successfully
dataFull.isnull().sum()[dataFull.isnull().sum()>0].sort_values(ascending = False)




# Categorical features : replace with the mode (most frequently occured value)
for col in ['MSZoning','Functional','Utilities','KitchenQual','SaleType','Exterior2nd','Exterior1st','Electrical']:
    dataFull[col].fillna(dataFull[col].mode()[0],inplace= True)

# Check if missing values are imputed successfully
dataFull.isnull().sum()[dataFull.isnull().sum()>0].sort_values(ascending = False)




# Impute features with more than five missing values

# Categorical features : replace all with 'None'
for col in ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageQual','GarageCond','GarageFinish','GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1','MasVnrType']:
    dataFull[col].fillna('None',inplace = True)

# Check if missing values are imputed successfully
dataFull.isnull().sum()[dataFull.isnull().sum()>0].sort_values(ascending = False)




dataFull['MasVnrArea'].fillna(dataFull['MasVnrArea'].mean(), inplace=True)

# Check if missing values are imputed successfully
dataFull.isnull().sum()[dataFull.isnull().sum()>0].sort_values(ascending = False)




# Cut 'YearBuilt' into 10 parts
dataFull['YearBuiltCut'] = pd.qcut(dataFull.YearBuilt,10)
# Impute the missing values of 'GarageYrBlt' based on the median of 'YearBuilt' 
dataFull['GarageYrBlt']= dataFull.groupby(['YearBuiltCut'])['GarageYrBlt'].transform(lambda x : x.fillna(x.median()))
# convert the values to integers
dataFull['GarageYrBlt'] = dataFull['GarageYrBlt'].astype(int)
# Drop 'YearBuiltCut' column
dataFull.drop('YearBuiltCut',axis=1,inplace=True)
# Check if missing values are imputed successfully
dataFull.isnull().sum()[dataFull.isnull().sum()>0].sort_values(ascending = False)




# Cut 'LotArea' into 10 parts
dataFull['LotAreaCut'] = pd.qcut(dataFull.LotArea,10)
# Impute the missing values of 'LotFrontage' based on the median of 'LotArea' and 'Neighborhood'
dataFull['LotFrontage']= dataFull.groupby(['LotAreaCut','Neighborhood'])['LotFrontage'].transform(lambda x : x.fillna(x.median()))
dataFull['LotFrontage']= dataFull.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x : x.fillna(x.median()))
# Drop 'LotAreaCut' column
dataFull.drop('LotAreaCut',axis=1,inplace=True)

# Check if missing values are imputed successfully
dataFull.isnull().sum()[dataFull.isnull().sum()>0].sort_values(ascending = False)




get_ipython().run_cell_magic('HTML', '', '<h1>Correcting Features</h1>')




dataFull.select_dtypes(include=[np.number]).columns




# Converting numeric features to categorical features
strCols = ['YrSold','YearRemodAdd','YearBuilt','MoSold','MSSubClass','GarageYrBlt']
for i in strCols:
    dataFull[i]=dataFull[i].astype(str)




get_ipython().run_cell_magic('HTML', '', '<h1>Adding Features</h1>')




dataFull.select_dtypes(include=[np.object]).columns




dataFull["oExterQual"] = dataFull.ExterQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
dataFull["oBsmtQual"] = dataFull.BsmtQual.map({'None':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
dataFull["oBsmtExposure"] = dataFull.BsmtExposure.map({'None':1, 'No':2, 'Av':3, 'Mn':3, 'Gd':4})
dataFull["oHeatingQC"] = dataFull.HeatingQC.map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
dataFull["oKitchenQual"] = dataFull.KitchenQual.map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4})
dataFull["oFireplaceQu"] = dataFull.FireplaceQu.map({'None':1, 'Po':2, 'Fa':3, 'TA':4, 'Gd':5, 'Ex':6})
dataFull["oGarageFinish"] = dataFull.GarageFinish.map({'None':1, 'Unf':2, 'RFn':3, 'Fin':4})
dataFull["oPavedDrive"] = dataFull.PavedDrive.map({'N':1, 'P':2, 'Y':3})




dataFull.select_dtypes(include=[np.number]).columns




dataFull['HouseSF'] = dataFull['1stFlrSF'] + dataFull['2ndFlrSF'] + dataFull['TotalBsmtSF']
dataFull['PorchSF'] = dataFull['3SsnPorch'] + dataFull['EnclosedPorch'] + dataFull['OpenPorchSF'] + dataFull['ScreenPorch']
dataFull['TotalSF'] = dataFull['HouseSF'] + dataFull['PorchSF'] + dataFull['GarageArea']




get_ipython().run_cell_magic('HTML', '', '<h1>Skewness and Kurtosis</h1>')




# Estimate Skewness and Kurtosis of the data
trainData.skew(), trainData.kurt()




# Plot the Skewness of the data
sns.distplot(trainData.skew(),axlabel ='Skewness')




# Plot the Kurtosis of the data
sns.distplot(trainData.kurt(),axlabel ='Kurtosis')




get_ipython().run_cell_magic('HTML', '', '<h1>Label Encoding</h1>')




from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder, Imputer
from scipy.stats import skew

# Label encoding class
class labenc(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        label = LabelEncoder()
        X['YrSold']=label.fit_transform(X['YrSold'])
        X['YearRemodAdd']=label.fit_transform(X['YearRemodAdd'])
        X['YearBuilt']=label.fit_transform(X['YearBuilt'])
        X['MoSold']=label.fit_transform(X['MoSold'])
        X['GarageYrBlt']=label.fit_transform(X['GarageYrBlt'])
        return X
    
# Skewness transform class
class skewness(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        skewness = X.select_dtypes(include=[np.number]).apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= 1].index
        X[skewness_features] = np.log1p(X[skewness_features])
        return X

# One hot encoding class
class onehotenc(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X = pd.get_dummies(X)
        return X




# Creating a copy of the full dataset
dataFullCopy = dataFull.copy()

# Creating a new fata with aplied transformations using sklearn Pipeline
from sklearn.pipeline import Pipeline
dataPipeline = pipeline.fit_transform(dataFullCopy)
dataFull.shape, dataPipeline.shape




dataPipeline.head()





X_train = dataPipeline[:trainData.shape[0]]
y_train = X_train['SalePrice']
X_train.drop(columns = 'SalePrice', inplace=True)
X_test = dataPipeline[trainData.shape[0]:]
X_test.drop(columns = 'SalePrice', inplace=True)
X_train.shape, y_train.shape, X_test.shape




get_ipython().run_cell_magic('HTML', '', '<h1>Transformation and Scaling</h1>')




# SalesPrices plot with three different fitted distributions
plt.figure(2); plt.title('Normal')
sns.distplot(y_train, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y_train, kde=False, fit=st.lognorm)
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y_train, kde=False, fit=st.johnsonsu)




# transforming 'SalePrice' into normal distribution
y_train_transformed = np.log(y_train)
y_train_transformed.skew(), y_train_transformed.kurt()




# plotting 'SalePrice' before and after the transformation
plt.figure(1); plt.title('Before transformation')
sns.distplot(y_train)
plt.figure(2); plt.title('After transformation')
sns.distplot(y_train_transformed)




# Using Robust Scaler to transform X_train
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit(X_train).transform(X_train)
X_test_scaled = robust_scaler.transform(X_test)




# Shape of final data we will be working on
X_train_scaled.shape, y_train_transformed.shape, X_test_scaled.shape




get_ipython().run_cell_magic('HTML', '', '<h1>Feature Selection</h1>')




# Display features by their importance (lasso regression coefficient)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.001)
lasso.fit(X_train_scaled,y_train_transformed)
y_pred_lasso = lasso.predict(X_test_scaled)
lassoCoeff = pd.DataFrame({"Feature Importance":lasso.coef_}, index=dataPipeline.drop(columns = 'SalePrice').columns)
lassoCoeff.sort_values("Feature Importance",ascending=False)




# Plot features by importance (feature coefficient in the model)
lassoCoeff[lassoCoeff["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(20,35))




get_ipython().run_cell_magic('HTML', '', '<h1>Principal Component Analysis</h1>')




from sklearn.decomposition import PCA
# Concatenate the training and test datasets into a single datafram
dataFull2 = np.concatenate([X_train_scaled,X_test_scaled])
# Choose the number of principle components such that 95% of the variance is retained
pca = PCA(0.95)
dataFull2 = pca.fit_transform(dataFull2)
varPCA = np.round(pca.explained_variance_ratio_*100, decimals = 1)
# Principal Component Analysis of data
print(varPCA)




# Principal Component Analysis plot of the data
plt.figure(figsize=(16,12))
plt.bar(x=range(1,len(varPCA)+1), height = varPCA)
plt.ylabel("Explained Variance (%)", size = 15)
plt.xlabel("Principle Components", size = 15)
plt.title("Principle Component Analysis Plot : Training Data", size = 15)
plt.show()




# Shape of final data we will be working on
X_train_scaled = dataFull2[:trainData.shape[0]]
X_test_scaled = dataFull2[trainData.shape[0]:]
X_train_scaled.shape, y_train_transformed.shape, X_test_scaled.shape




get_ipython().run_cell_magic('HTML', '', '<h1>Testing Different Models</h1>')




# importing the models
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, Lasso, SGDRegressor, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import LinearSVR,SVR
# creating the models
models = [
             LinearRegression(),
             SVR(),
             SGDRegressor(),
             SGDRegressor(max_iter=1000, tol = 1e-3),
             GradientBoostingRegressor(),
             RandomForestRegressor(),
             Lasso(),
             Lasso(alpha=0.01,max_iter=10000),
             Ridge(),
             BayesianRidge(),
             KernelRidge(),
             KernelRidge(alpha=0.6,kernel='polynomial',degree = 2,coef0=2.5),
             ElasticNet(),
             ElasticNet(alpha = 0.001,max_iter=10000),    
             ExtraTreesRegressor(),
             ]

names = ['Linear regression','Support vector regression','Stochastic gradient descent','Stochastic gradient descent 2','Gradient boosting tree','Random forest','Lasso regression','Lasso regression 2','Ridge regression','Bayesian ridge regression','Kernel ridge regression','Kernel ridge regression 2','Elastic net regularization','Elastic net regularization 2','Extra trees regression']





# Define a root mean square error function
def rmse(model,X,y):
    rmse = np.sqrt(-cross_val_score(model,X,y,scoring="neg_mean_squared_error",cv=5))
    return rmse




from sklearn.model_selection import KFold,cross_val_score
warnings.filterwarnings('ignore')

# Perform 5-folds cross-calidation to evaluate the models 
for model, name in zip(models, names):
    # Root mean square error
    score = rmse(model,X_train_scaled,y_train_transformed)
    print("- {} : mean : {:.6f}, std : {:4f}".format(name, score.mean(),score.std()))




get_ipython().run_cell_magic('HTML', '', '<h1>Hyper-parameter Tuning</h1>')




from sklearn.model_selection import GridSearchCV

class gridSearch():
    def __init__(self,model):
        self.model = model
    def grid_get(self,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5,scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled,y_train_transformed)
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
        print('\nBest parameters : {}, best score : {}'.format(grid_search.best_params_,np.sqrt(-grid_search.best_score_)))




gridSearch(KernelRidge()).grid_get(
        {'alpha':[3.5,4,4.5,5,5.5,6,6.5], 'kernel':["polynomial"], 'degree':[3],'coef0':[1,1.5,2,2.5,3,3.5]})




gridSearch(ElasticNet()).grid_get(
        {'alpha':[0.006,0.0065,0.007,0.0075,0.008],'l1_ratio':[0.070,0.075,0.080,0.085,0.09,0.095],'max_iter':[10000]})




gridSearch(Ridge()).grid_get(
        {'alpha':[10,20,25,30,35,40,45,50,55,57,60,65,70,75,80,100],'max_iter':[10000]})




gridSearch(SVR()).grid_get(
        {'C':[13,15,17,19,21],'kernel':["rbf"],"gamma":[0.0005,0.001,0.002,0.01],"epsilon":[0.01,0.02,0.03,0.1]})




gridSearch(Lasso()).grid_get(
       {'alpha':[0.01,0.001,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009],'max_iter':[10000]})

We see that the models perform almost the same way with a score of 0.116. Let's define these models with the their respective best hyper-parameters.


lasso = Lasso(alpha= 0.0006, max_iter= 10000)
ridge = Ridge(alpha=35, max_iter= 10000)
svr = SVR(C = 13, epsilon= 0.03, gamma = 0.001, kernel = 'rbf')
ker = KernelRidge(alpha=6.5 ,kernel='polynomial', degree=3 , coef0=2.5)
ela = ElasticNet(alpha=0.007,l1_ratio=0.07,max_iter=10000)
bay = BayesianRidge()




get_ipython().run_cell_magic('HTML', '', '<h1>Combining Models</h1>')




from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor




# Creating the stacking function
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        kff = KFold(n_splits=5, random_state=42, shuffle=True)
        self.kf = kff
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean




# Impute the training dataset
X_scaled_imputed = Imputer().fit_transform(X_train_scaled)
y_log_imputed = Imputer().fit_transform(y_train_transformed.values.reshape(-1,1)).ravel()




X_scaled_imputed.shape,y_log_imputed.shape,X_test_scaled.shape




# Calculating the score
stack_model = stacking(mod=[lasso,ridge,svr,ker,ela,bay],meta_model=ker)
score = rmse(stack_model,X_scaled_imputed,y_log_imputed)
print(score.mean())




# Combining the extracted features generated from stacking whith original features
X_train_stack,X_test_stack = stack_model.get_oof(X_scaled_imputed,y_log_imputed,X_test_scaled)
X_train_add = np.hstack((X_scaled_imputed,X_train_stack))
X_test_add = np.hstack((X_test_scaled,X_test_stack))
X_train_add.shape,X_test_add.shape




# Calculate the final score of the model
score = rmse(stack_model,X_train_add,y_log_imputed)
print(score.mean())




get_ipython().run_cell_magic('HTML', '', '<h1>Making Predictions</h1>')




# Fit the model to the dataset generated with stacking
stack_model.fit(X_train_add,y_log_imputed)




# Making prediction on the test set generated by stacking
predicted_prices = np.exp(stack_model.predict(X_test_add))
# Prepare the csv file
my_submission = pd.DataFrame({'Id': testData.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)




get_ipython().run_cell_magic('HTML', '', '<h2>Thanks for reading my notebook. If you liked my kernel please kindly UPVOTE for other people to see. If you have any remarks, please leave a comment bellow.<h2>')






