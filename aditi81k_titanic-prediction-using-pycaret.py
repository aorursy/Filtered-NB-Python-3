#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd 




get_ipython().system('pip3 install pycaret')




train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")




from pycaret import classification
classification_setup = classification.setup(data=train,target='Survived', ignore_features = ['Ticket', 'Name', 'PassengerId'], silent = True, session_id=42)




classification.compare_models()




from pycaret.classification import *
models()
compare_models(whitelist = models(type='ensemble').index.tolist())




lgb_classifier = classification.create_model('lightgbm')




params = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
          'n_estimators':[100,250,500,750,1000,1250,1500,1750],
          'max_depth': np.random.randint(1, (len(train.columns)*.85),20),
          'max_features': np.random.randint(1, len(train.columns),20),
          'min_samples_split':[2,4,6,8,10,20,40,60,100], 
          'min_samples_leaf':[1,3,5,7,9],
          'criterion': ["gini", "entropy"]}

tune_lgb = classification.tune_model(lgb_classifier, custom_grid = params)




# Tune the model
params = {'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
tune_ridge = classification.tune_model(create_model('ridge'), custom_grid = params, n_iter=50, fold=50)




# ensemble boosting
bagging = classification.ensemble_model(tune_lgb, method= 'Bagging')




from pycaret.classification import blend_models
# blending all models
blend_all = blend_models(method='hard')




# create individual models for stacking
ridge_cls = classification.create_model('ridge')
extre_tr = classification.create_model('et')
lgb = classification.create_model('lightgbm')
cat_cls = classification.create_model('catboost')
lg_cls = classification.create_model('lr')




from pycaret.classification import stack_models
# stacking models
stacker = stack_models(estimator_list = [ridge_cls, extre_tr, lgb, cat_cls, lg_cls],method='hard')




interpret_model(tune_lgb)




from pycaret.classification import *


y_pred = predict_model(tune_lgb, data=test)




y_pred




submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred['Label']
    })
submission.to_csv("submission.csv", index=False)






