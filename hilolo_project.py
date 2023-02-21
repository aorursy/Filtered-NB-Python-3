#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))

df = pd.read_csv('../input/data.txt', delimiter=' '); 
print(df.shape)

# MEHDI EL KRAIMI 
# MEHDI EL AISSI





class_counts = df.groupby('CARVP').size()
print(class_counts)




types = df.dtypes
print(types)




from pandas import set_option
set_option('display.width', 80)
set_option('precision', 2)
description = df.loc[:, df.columns != 'ID'].describe()
print(description)





set_option('display.width', 80)
set_option('precision', 3)
correlations = df.loc[:, df.columns != 'ID'].corr(method='pearson')
print(correlations)




skew = df.loc[:, df.columns != 'ID'].skew()
print(skew)





from matplotlib import pyplot
df.loc[:, df.columns != 'ID'].plot(kind='box', subplots=True, layout=(10,4), sharex=False, sharey=False,figsize=(8,50))
pyplot.show()




from matplotlib import pyplot
import numpy
correlations = df.loc[:, df.columns != 'ID'].corr()
#  correlation matrix
fig = pyplot.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,30,1)

ax.set_yticks(ticks)
ax.set_yticklabels(df.columns.values.tolist())
pyplot.show()




from matplotlib import pyplot
df.loc[:, df.columns != 'ID'].plot(kind='density', subplots=True, layout=(10,4), sharex=False,figsize=(20,8))
pyplot.show()




from matplotlib import pyplot
df.hist(figsize=(20,20))
pyplot.show()




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# variable contenant la liste des noms de variables
var=df.columns

# liste des variables explicatives quantitatives

varquant=var[1:31]

# liste des variables explicatives qualitatives
varqual=var[31:56]





#transformation en matrice numpy - seul reconnu par scikit-learn
df = df.as_matrix()

#X matrice des var. explicatives quantitatives
X_quant = df[:,1:31]
#X matrice des var. explicatives qualitatives
X_quali = df[:,31:56]

#y vecteur de la var. à prédire
y = df[:,56]

#utilisation du module model_selection de scikit-learn (sklearn)
from sklearn.cross_validation import train_test_split
#subdivision des données 
X_train, X_test, y_train, y_test = train_test_split(X_quant, y, test_size = 0.3, random_state = 0)




from sklearn.linear_model import LogisticRegression
#création d'une instance de la classe
lr = LogisticRegression()
from sklearn.feature_selection import RFE
selecteur = RFE(estimator = lr)

sol = selecteur.fit(X_train,y_train)
print(X_train.shape)







#nombre de var. sélectionnées
print(sol.n_features_) 




#liste des variables sélectionnées
print(sol.support_)





#ordre de suppression
print(sol.ranking_)




from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as AA
classifier = AA()
classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypred)
print(cm)





accuracy1 = ((201+65)/(201+23+65+30)) * 100
print(accuracy1)




#Applying Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

ypred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypred)
print(cm)





accuracy2 = ((196+66)/(196+28+29+66)) * 100
print(accuracy2)




#Applying Logistic Regression Algorithm
from sklearn import tree as XXA
classifier = XXA.DecisionTreeClassifier()
classifier.fit(X_train,y_train)

ypred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypred)
print(cm)




accuracy3 = ((203 +67)/(203+21+67+28)) * 100
print(accuracy3)




from sklearn import svm
classifier = svm.LinearSVC()
classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypred)
print(cm)




accuracy4 = ((107 +68)/(107+117+68+27)) * 100
print(accuracy4)




from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier( random_state=1)
classifier.fit(X_train,y_train)

ypred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, ypred)
print(cm)




accuracy5 = ((101 +86)/(101+123+9+86)) * 100
print(accuracy5)




AQ1 = int(accuracy1)
AQ2 = int(accuracy2)
AQ3 = int(accuracy3)
AQ4 = int(accuracy4)
AQ5 = int(accuracy5)

objects = ('Analyse discriminante ', 'Régresion logistique', 'Arbre de classfication', 'SVM', 'Réseau de neurones')
y_pos = np.arange(len(objects))
performance = [AQ1,AQ2,AQ3,AQ4,AQ5]
 
plt.scatter(y_pos, performance, alpha=1)
plt.plot(y_pos, performance,color='green')
plt.xticks(y_pos, objects)
plt.ylabel('Précision %')
plt.xticks(rotation=45)
plt.title('Algorithm Précision')
plt.show()




AQ10 = int(100-accuracy1)
AQ20 = int(100-accuracy2)
AQ30 = int(100-accuracy3)
AQ40 = int(100-accuracy4)
AQ50 = int(100-accuracy5)

objects = ('Analyse discriminante ', 'Régresion logistique', 'Arbre de classfication', 'SVM', 'Réseau de neurones')
y_pos = np.arange(len(objects))
performance = [AQ10,AQ20,AQ30,AQ40,AQ50]
 
plt.scatter(y_pos, performance, alpha=1)
plt.plot(y_pos, performance,color='red')
plt.xticks(y_pos, objects)
plt.ylabel('mal classé %')
plt.xticks(rotation=45)
plt.title('Algorithm mal classé')
plt.show()

