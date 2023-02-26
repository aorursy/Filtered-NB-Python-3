#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




df = pd.read_csv('../input/titanic_data.csv')
print(df.shape)
print(df.head())




get_ipython().set_next_input('Q: How do you access row 55-60 with Names, Survived, Age');get_ipython().run_line_magic('pinfo', 'Age')




print(df[['Name','Survived','Age']].values[55:61])




df = df.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1)
df = df.dropna()
print(df.shape)
print(df.head())




sexConvDict = {"male":1 ,"female" :2}
df['Sex'] = df['Sex'].apply(sexConvDict.get).astype(int)




# **Decision Tree**
# no need standardisation/normalisation




from sklearn import tree
from sklearn.model_selection import train_test_split

#features = ['Sex'] #0.787
#features = ['Sex','Pclass'] #0.787
features = ['Sex','Pclass','Age'] #0.807
#features = ['Sex','Pclass','Age','Fare'] #0.737
#features = ['Sex','Pclass','Age','Fare','SibSp'] #0.762
#features = ['Sex','Pclass','Age','Fare','SibSp','Parch'] #0.759
X = df[features].values
y = df['Survived'].values

#Not standardised
print(X[0])




#sklearn returns Python array not Numpy/Pandas

#Split 50% to Train, 50% to Test randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=1)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)




result = len([n for n in y_test ^ y_predict if n==1])
print(result/len(y_predict))




from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)




# Accuracy score




from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)




# Normalisation




from sklearn.preprocessing import StandardScaler

#features = ['Sex','Pclass','Age'] #0.779 #0.787
#features = ['Sex','Pclass','Age','Fare'] #0.796 #0.807
features = ['Sex','Pclass','Age','Fare','SibSp','Parch'] #0.838 #0.801

scaler = StandardScaler()
X_standard = scaler.fit_transform(df[features].values)

X_train, X_test, X_std_train, X_std_test, y_train, y_test = train_test_split(X, X_standard, y, test_size=0.50, random_state=1)

#Standardised
print(X_standard[0])




from sklearn.neural_network import MLPClassifier
clf = MLPClassifier()
#clf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(3, 2), random_state=0)
clf = clf.fit(X_std_train, y_train)
y_predict =clf.predict(X_std_test)




confusion_matrix(y_test, y_predict)




accuracy_score(y_test, y_predict)




from sklearn import svm

#features = ['Sex','Pclass','Age'] #0.807
features = ['Sex','Pclass','Age','Fare','SibSp','Parch'] #0.807

clf = svm.SVC()
clf = clf.fit(X_std_train, y_train)
y_predict = clf.predict(X_std_test)




# Confusion Matrix




confusion_matrix(y_test, y_predict)




accuracy_score(y_test, y_predict)

