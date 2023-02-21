#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# eklediğim kütüphaneler (buraya eklenecek)
from sklearn import preprocessing
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn import ensemble

from sklearn import utils
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




data = pd.read_csv('../input/bank-customer-churn-modeling/Churn_Modelling.csv') # datasetini pandas ile okumak için yolu belirtiyoruz




data.head(10) # görmek amaçlı baştan 10 elemana bakıyoruz(kontrol amaçlı bir nedeni yok)




data.info() # satır ve sütunların eskiklik ve doluluk oranlarına bakıyoruz(varsa ve fazla ise dropout yapabiliriz)




kayip_degerler = data.isnull().sum() # kayıp boş değer var mı kontrolü yapıyoruz(aşağıda görüldüğü gibi böyle bir değer yok)
kayip_degerler[0:14] # öznitelik sayısı kadar olan kolonlara baktık




# balance'ı kullanmayı düşündüğümden normalize ediyorum ve yeni bir kolon olarak veri setime ekliyorum
x = data[['Balance']].values.astype('float64')
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data['Balance_Norm'] = pd.DataFrame(x_scaled)
# estimated salary
y = data[['EstimatedSalary']].values.astype('float64')
y_scaled = min_max_scaler.fit_transform(y)
data['EstimatedSalary_Norm'] = pd.DataFrame(y_scaled)
# credit score
z = data[['CreditScore']].values.astype('float64')
z_scaled = min_max_scaler.fit_transform(z)
data['CreditScore_Norm'] = pd.DataFrame(z_scaled)
data.head(7)




secilen_oznitelik = ['Balance_Norm','EstimatedSalary_Norm','CreditScore_Norm']
secilen_kolon = data[secilen_oznitelik]
secilen_kolon.head(10)




tahmin_kolonu = data.Exited
tahmin_kolonu.describe()




secilen_kolon_train,secilen_kolon_test,tahmin_kolonu_train,tahmin_kolonu_test = train_test_split(secilen_kolon,tahmin_kolonu,test_size=0.2,random_state=0)




regressor = DecisionTreeClassifier(random_state = 0, max_depth=5) #Decision tree regresyon gerçekleştirimi.
regressor.fit(secilen_kolon_train,tahmin_kolonu_train)
tahmin_kolonu_pred_dt = regressor.predict(secilen_kolon_test)




exited_tahmini_dt = regressor.predict(secilen_kolon_test) #Tahmin ettiğimiz değerlerle gerçek değer ne kadar yakın ?
mean_absolute_error(tahmin_kolonu_test,exited_tahmini_dt) #Tahmin ne kadar yakınsa fark 0'a o kadar yakın olur.




print('confusion matrix:\n', confusion_matrix(tahmin_kolonu_test,tahmin_kolonu_pred_dt)) 
print("recall score: ", recall_score(tahmin_kolonu_test,tahmin_kolonu_pred_dt))
print("precision score ", precision_score(tahmin_kolonu_test,tahmin_kolonu_pred_dt))
print("f1 score: ", f1_score(tahmin_kolonu_test,tahmin_kolonu_pred_dt))
print("accuary score: ", accuracy_score(tahmin_kolonu_test,tahmin_kolonu_pred_dt))




forest_model = RandomForestClassifier(random_state = 0,n_estimators=250,max_depth=6) #Random Forest regresyon gerçekleşimi
forest_model.fit(secilen_kolon_train,tahmin_kolonu_train)
tahmin_kolonu_pred_rf = forest_model.predict(secilen_kolon_test)




exited_tahmini_rf = forest_model.predict(secilen_kolon_test)#Tahmin ettiğimiz değerlerle gerçek değer ne kadar yakın ?
mean_absolute_error(tahmin_kolonu_test,exited_tahmini_rf)#Tahmin ne kadar yakınsa fark 0'a o kadar yakın olur




print('confusion matrix:\n', confusion_matrix(tahmin_kolonu_test,tahmin_kolonu_pred_rf)) 
print("recall score: ", recall_score(tahmin_kolonu_test,tahmin_kolonu_pred_rf))
print("precision score ", precision_score(tahmin_kolonu_test,tahmin_kolonu_pred_rf))
print("f1 score: ", f1_score(tahmin_kolonu_test,tahmin_kolonu_pred_rf))
print("accuary score: ", accuracy_score(tahmin_kolonu_test,tahmin_kolonu_pred_rf))




gradient_model = GradientBoostingClassifier(random_state = 0, n_estimators=100,learning_rate=0.01,max_depth=6)
gradient_model.fit(secilen_kolon_train,tahmin_kolonu_train)
tahmin_kolonu_pred_gb = gradient_model.predict(secilen_kolon_test) 




exited_tahmini_gb = gradient_model.predict(secilen_kolon_test)
mean_absolute_error(tahmin_kolonu_test,exited_tahmini_gb)




print('confusion matrix:\n', confusion_matrix(tahmin_kolonu_test,tahmin_kolonu_pred_gb)) 
print("recall score: ", recall_score(tahmin_kolonu_test,tahmin_kolonu_pred_gb))
print("precision score ", precision_score(tahmin_kolonu_test,tahmin_kolonu_pred_gb))
print("f1 score: ", f1_score(tahmin_kolonu_test,tahmin_kolonu_pred_gb))
print("accuary score: ", accuracy_score(tahmin_kolonu_test,tahmin_kolonu_pred_gb))




kneighbors_model = KNeighborsClassifier(n_neighbors=15)
kneighbors_model.fit(secilen_kolon_train,tahmin_kolonu_train)
tahmin_kolonu_pred_kn = kneighbors_model.predict(secilen_kolon_test)




exited_tahmini_kn = kneighbors_model.predict(secilen_kolon_test)
mean_absolute_error(tahmin_kolonu_test,exited_tahmini_kn)




print('confusion matrix:\n', confusion_matrix(tahmin_kolonu_test,tahmin_kolonu_pred_kn)) 
print("recall score: ", recall_score(tahmin_kolonu_test,tahmin_kolonu_pred_kn))
print("precision score ", precision_score(tahmin_kolonu_test,tahmin_kolonu_pred_kn))
print("f1 score: ", f1_score(tahmin_kolonu_test,tahmin_kolonu_pred_kn))
print("accuary score: ", accuracy_score(tahmin_kolonu_test,tahmin_kolonu_pred_kn))




svm_model = SVC(degree=3,probability=True,cache_size=100,kernel='linear')
svm_model.fit(secilen_kolon_train,tahmin_kolonu_train)
tahmin_kolonu_pred_svm = svm_model.predict(secilen_kolon_test)




exited_tahmini_svm = svm_model.predict(secilen_kolon_test)
mean_absolute_error(tahmin_kolonu_test,exited_tahmini_svm)




print('confusion matrix:\n', confusion_matrix(tahmin_kolonu_test,tahmin_kolonu_pred_svm)) 
print("recall score: ", recall_score(tahmin_kolonu_test,tahmin_kolonu_pred_svm))
print("precision score ", precision_score(tahmin_kolonu_test,tahmin_kolonu_pred_svm))
print("f1 score: ", f1_score(tahmin_kolonu_test,tahmin_kolonu_pred_svm))
print("accuary score: ", accuracy_score(tahmin_kolonu_test,tahmin_kolonu_pred_svm))




n_bayes_model = MultinomialNB()
n_bayes_model.fit(secilen_kolon_train,tahmin_kolonu_train)
tahmin_kolonu_pred_nb = n_bayes_model.predict(secilen_kolon_test)




exited_tahmini_nb = n_bayes_model.predict(secilen_kolon_test)
mean_absolute_error(tahmin_kolonu_test,exited_tahmini_nb)




print('confusion matrix:\n', confusion_matrix(tahmin_kolonu_test,tahmin_kolonu_pred_nb)) 
print("recall score: ", recall_score(tahmin_kolonu_test,tahmin_kolonu_pred_nb))
print("precision score ", precision_score(tahmin_kolonu_test,tahmin_kolonu_pred_nb))
print("f1 score: ", f1_score(tahmin_kolonu_test,tahmin_kolonu_pred_nb))
print("accuary score: ", accuracy_score(tahmin_kolonu_test,tahmin_kolonu_pred_nb))
   




estimators =[('kneighbors_model', kneighbors_model),('gradient_model',gradient_model),('forest_model',forest_model),('n_bayes_model',n_bayes_model),('svm_model',svm_model)]
ensemble = VotingClassifier(estimators,voting='soft',weights=[1,2,2,1,2])
ensemble.fit(secilen_kolon_train,tahmin_kolonu_train)
ensemble.score(secilen_kolon_test,tahmin_kolonu_test)




estimators =[('kneighbors_model', kneighbors_model),('gradient_model',gradient_model),('forest_model',forest_model),('n_bayes_model',n_bayes_model),('svm_model',svm_model)]
ensemble = VotingClassifier(estimators,voting='hard')
ensemble.fit(secilen_kolon_train,tahmin_kolonu_train)
ensemble.score(secilen_kolon_test,tahmin_kolonu_test)

