#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from nolearn.dbn import DBN

import timeit









train = pd.read_csv("../input/train.csv")
features = train.columns[1:]
X = train[features]
y = train['label']









print X.shape
print y.shape




y.shape




X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.01,random_state=42)




X_train.shape




X_test.shape




#******************************************RANDOM FOREST CLASSIFIER***********************************




randomForest = RandomForestClassifier()
randomForest.fit(X_train,y_train)




y_pred = randomForest.predict(X_test)




score = accuracy_score(y_test,y_pred)
print "Random Forest accuraccy socre :  %s" %(score)




#Import Test data and create kaggle submission file 




test_data = pd.read_csv("../input/test.csv")




test_data.shape




test_data.head()




test_prediction_class = randomForest.predict(test_data)




ImageIdList = range(1,28001)




pd.DataFrame({'ImageId':ImageIdList,'Label':test_prediction_class}).to_csv('random_forest_submission.csv',index=None)




#*************************Stochastic Gradient Descent**********************




clfSGD = SGDClassifier()
clfSGD.fit(X_train,y_train)
get_ipython().run_line_magic('time', 'clfSGDPredict = clfSGD.predict(X_test)')
sgd_score = accuracy_score(y_test,clfSGDPredict)
print "SGD accuracy score ",sgd_score




#***********Linear SVM******************




clfLinearSVM = LinearSVC()
clfLinearSVM.fit(X_train,y_train)
get_ipython().run_line_magic('time', 'clfLinearSVMPredict  = clfLinearSVM.predict(X_test)')
clfLinearSVMScore = accuracy_score(y_test,clfLinearSVMPredict)
print "Linear SVM Accuraccy score",clfLinearSVMScore




#**************KNeighborsClassifier***********************




knnClassifier = KNeighborsClassifier()




knnClassifier.fit(X_train,y_train)




knn_predict = knnClassifier.predict(X_test)
knn_score = accuracy_score(y_test,knn_predict)
print "knn accuraccy score",knn_score




get_ipython().run_line_magic('time', 'KnnPredictTEstData = knnClassifier.predict(test_data)')
KnnPredictTEstData




pd.DataFrame({'ImageId':ImageIdList,'Label':KnnPredictTEstData}).to_csv('knn_classifier_submission.csv',index=False)

