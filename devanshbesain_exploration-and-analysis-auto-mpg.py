#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory on kaggle.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))




data = pd.read_csv('../input/auto-mpg.csv',index_col='car name')




print(data.head())
print(data.index)
print(data.columns)




data.shape




data.isnull().any()




data.dtypes




data.horsepower.unique()




data = data[data.horsepower != '?']




print('?' in data.horsepower)




data.shape




data.dtypes




data.horsepower = data.horsepower.astype('float')
data.dtypes





data.describe()




data.mpg.describe()




sns.distplot(data['mpg'])




print("Skewness: %f" % data['mpg'].skew())
print("Kurtosis: %f" % data['mpg'].kurt())




def scale(a):
    b = (a-a.min())/(a.max()-a.min())
    return b




data_scale = data.copy()




data_scale ['displacement'] = scale(data_scale['displacement'])
data_scale['horsepower'] = scale(data_scale['horsepower'])
data_scale ['acceleration'] = scale(data_scale['acceleration'])
data_scale ['weight'] = scale(data_scale['weight'])
data_scale['mpg'] = scale(data_scale['mpg'])




data_scale.head()




data['Country_code'] = data.origin.replace([1,2,3],['USA','Europe','Japan'])
data_scale['Country_code'] = data.origin.replace([1,2,3],['USA','Europe','Japan'])




data_scale.head()




var = 'Country_code'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)




var = 'model year'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)




var = 'cylinders'
data_plt = pd.concat([data_scale['mpg'], data_scale[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)




corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);




factors = ['cylinders','displacement','horsepower','acceleration','weight','mpg']
corrmat = data[factors].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);




#scatterplot
sns.set()
sns.pairplot(data, size = 2.0,hue ='Country_code')
plt.show()




data.index




data[data.index.str.contains('subaru')].index.str.replace('(.*)', 'subaru dl')




data['Company_Name'] = data.index.str.extract('(^.*?)\s')




data['Company_Name'] = data['Company_Name'].replace(['volkswagen','vokswagen','vw'],'VW')
data['Company_Name'] = data['Company_Name'].replace('maxda','mazda')
data['Company_Name'] = data['Company_Name'].replace('toyouta','toyota')
data['Company_Name'] = data['Company_Name'].replace('mercedes','mercedes-benz')
data['Company_Name'] = data['Company_Name'].replace('nissan','datsun')
data['Company_Name'] = data['Company_Name'].replace('capri','ford')
data['Company_Name'] = data['Company_Name'].replace(['chevroelt','chevy'],'chevrolet')
data['Company_Name'].fillna(value = 'subaru',inplace=True)  ## Strin methords will not work on null values so we use fillna()




var = 'Company_Name'
data_plt = pd.concat([data_scale['mpg'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(20,10))
fig = sns.boxplot(x=var, y="mpg", data=data_plt)
fig.set_xticklabels(ax.get_xticklabels(),rotation=30)
fig.axis(ymin=0, ymax=1)
plt.axhline(data_scale.mpg.mean(),color='r',linestyle='dashed',linewidth=2)




data.Company_Name.isnull().any()




var='mpg'
data[data[var]== data[var].min()]




data[data[var]== data[var].max()]




var='displacement'
data[data[var]== data[var].min()]




data[data[var]== data[var].max()]




var='horsepower'
data[data[var]== data[var].min()]




data[data[var]== data[var].max()]




var='weight'
data[data[var]== data[var].min()]




data[data[var]== data[var].max()]




var='acceleration'
data[data[var]== data[var].min()]




data[data[var]== data[var].max()]




var = 'horsepower'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))




var = 'displacement'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))




var = 'weight'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))




var = 'acceleration'
plot = sns.lmplot(var,'mpg',data=data,hue='Country_code')
plot.set(ylim = (0,50))




data['Power_to_weight'] = ((data.horsepower*0.7457)/data.weight)




data.sort_values(by='Power_to_weight',ascending=False ).head()




data.head()




from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold




factors = ['cylinders','displacement','horsepower','acceleration','weight','origin','model year']
X = pd.DataFrame(data[factors].copy())
y = data['mpg'].copy()




X = StandardScaler().fit_transform(X)




X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=324)
X_train.shape[0] == y_train.shape[0]




regressor = LinearRegression()




regressor.get_params()




regressor.fit(X_train,y_train)




y_predicted = regressor.predict(X_test)




rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
rmse




gb_regressor = GradientBoostingRegressor(n_estimators=4000)
gb_regressor.fit(X_train,y_train)




gb_regressor.get_params()




y_predicted_gbr = gb_regressor.predict(X_test)




rmse_bgr = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr))
rmse_bgr




fi= pd.Series(gb_regressor.feature_importances_,index=factors)
fi.plot.barh()




from sklearn.decomposition import PCA




pca = PCA(n_components=2)




pca.fit(data[factors])




pca.explained_variance_ratio_




pca1 = pca.components_[0]
pca2 = pca.components_[1]




transformed_data = pca.transform(data[factors])




pc1 = transformed_data[:,0]
pc2 = transformed_data[:,1]




plt.scatter(pc1,pc2)




c = pca.inverse_transform(transformed_data[(transformed_data[:,0]>0 )& (transformed_data[:,1]>250)])




factors




c




data[(data['model year'] == 70 )&( data.displacement>400)]




cv_sets = KFold(n_splits=10, shuffle= True,random_state=100)
params = {'n_estimators' : list(range(40,61)),
         'max_depth' : list(range(1,10)),
         'learning_rate' : [0.1,0.2,0.3] }
grid = GridSearchCV(gb_regressor, params,cv=cv_sets,n_jobs=4)




grid = grid.fit(X_train, y_train)




grid.best_estimator_




gb_regressor_t = grid.best_estimator_




gb_regressor_t.fit(X_train,y_train)




y_predicted_gbr_t = gb_regressor_t.predict(X_test)




rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted_gbr_t))
rmse




data.duplicated().any()






