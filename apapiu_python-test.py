#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




train.shape




test.shape




train.SalePrice.hist()




np.log(train.SalePrice).hist()




train["SalePrice"] = np.log(train["SalePrice"]) + 1




train.head(20)




train.isnull().values.sum(axis = 0)




pd.DataFrame(train.isnull().values.sum(axis = 0), index = train.columns)




A decent amount of NA's.




Now let's create dummy variables:




X_train = pd.get_dummies(train.loc[:, 'MSSubClass':'SaleCondition'])
y_train = train.SalePrice




X_train.shape




X_train.head(3)




dtrain = xgb.DMatrix(X_train, label = y_train)
param = {'max_depth':1, 'eta':0.3} #booster : "gblinear"}
model_xgb = xgb.cv(param, dtrain, num_boost_round=1000, early_stopping_rounds=10)




rmse_err = []
for i in range(1,13):
    param = {'max_depth':i}
    cv_error = xgb.cv(param, dtrain, num_boost_round=1000, early_stopping_rounds=10, seed=21)
    rmse_err.append(cv_error["test-rmse-mean"].min())




pd.Series(rmse_err, index = range(1,13)).plot()




param = {'max_depth':2, 'eta':0.3} #booster : "gblinear"}
cv_error = xgb.cv(param, dtrain, num_boost_round=1000, early_stopping_rounds=10, seed=21)




cv_error["test-rmse-mean"].min() #CV rmse




nrounds = cv_error.shape[0]




nrounds




param = {'max_depth':3, 'eta':0.3}
model = xgb.train(param, dtrain, num_boost_round=nrounds)




importance = model.get_fscore()
importance = pd.DataFrame(list(importance.values()), index = importance.keys(), columns = ["f_score"])




importance.sort("f_score", ascending = False).head(30)




train[["LotFrontage", "SalePrice"]].plot(x = "LotFrontage", y = "SalePrice", kind="scatter")




important_feats = importance.sort_values("f_score", ascending = False).head(4).index




important_feats




imp_train = train[important_feats]




imp_train["SalePrices"] = train["SalePrice"]




imp_train.head()




sns.pairplot(imp_train.dropna())




imp_train.LotArea = np.log(imp_train.LotArea)
imp_train.BsmtUnfSF = np.log(imp_train.LotArea)
imp_train.LotFrontage = np.log(imp_train.LotFrontage)
imp_train.SalePrices = np.log(imp_train.SalePrices)




sns.pairplot(imp_train.dropna())






