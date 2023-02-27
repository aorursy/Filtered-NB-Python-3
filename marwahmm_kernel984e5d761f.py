#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




titanic_train = pd.read_csv('../input/train.csv') 
titanic_test = pd.read_csv('../input/test.csv')




titanic_data= pd.concat([titanic_train,titanic_test])




titanic_data.head()




titanic_data.tail()




titanic_train.isnull().sum()




titanic_test.isnull().sum()




titanic_data.isnull().sum()




titanic_data.dtypes




import seaborn as sns
sns.heatmap(titanic_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.3) #data.corr()-->correlation matrix
fig=plt.titanic_data()
fig.set_size_inches(12,10)
fig.show()




#snsn.heatmap.(titanic_train.isnull(),yticklabels***False)
#import seaborn as sns 
#sns.heatmap(titanic_data.isnull(), xticklabels=True, yticklabels=True)
#titanic_data()




titanic_data["Survived"].fillna('0', inplace=True)
titanic_data["Fare"].fillna('0', inplace=True)
titanic_data["Cabin"].fillna('U', inplace=True)




mean = titanic_data['Age'].mean()




titanic_data["Age"].fillna(mean, inplace=True)




titanic_data.head()




titanic_data.tail()




titanic_data.Survived = titanic_data.Survived.apply(int)
titanic_data.Age = titanic_data.Age.apply(int)
titanic_data.Fare = titanic_data.Fare.apply(int)




titanic_data.dtypes




#titanic_data.dropna(inplace=True)




#sns.heatmap(titanic_data.isnull(), xticklabels=True, yticklabels=True)




# Data visulization: 




sns.boxplot(x ='Pclass', y= 'Age', data= titanic_data)




#number of passengers: 
print(" number of passengers in original data:" +str(len(titanic_data)))




import seaborn as sns
sns.countplot(x= "Survived", data= titanic_data)




sns.countplot(x= "Survived", hue="Sex", data= titanic_data)




# the number of survived passengers among different Pclasses:  
import seaborn as sns
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)




#histogram for age : 




import seaborn as sns
sns.countplot(x='Ticket', hue='Pclass', data=titanic_data)




#Defing the age ditribution : 
titanic_data["Fare"].plot.hist(bins=20 , figsize=(10,5)) 




sns.countplot(x= "SibSp", data= titanic_data)




#sns.countplot(x = "Cabin", data = titanic_data)




sns.countplot(x = "Parch", data = titanic_data)




titanic_data.head()




#dropping unneccary columns :
titanic_data.drop(['Ticket','Name','Cabin'], axis = 1, inplace = True) 




titanic_data.head()




titanic_data = pd.get_dummies(titanic_data , drop_first=True)
titanic_data.head()

# Train Data:
#1:Defining the predicted variable and the independed variables.
#2:splitting data into  training and testing .
#3:Creating a model.
#4:Predictions.
#5:Evaluating the performane of the model(classification report). 
#6:Testing the score accuracy.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 




#step1:
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




#step2: feature scaling :
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train) 
X_test = Sc_X.transform(X_test)

# X_train_mm.shape, X_test_mm.shape
#model = KNeighborsClassifier(n_neighbors=426, )
#model.fit(X_train_mm, y_train)
#model.score(X_test_mm, y_test)




len(y)




import math 
math.sqrt(len(y_test))




#step3: define the model using KNeigbors classifier: init K_NN:
model = KNeighborsClassifier(n_neighbors=19,p=2,)
model.fit(X_train, y_train)




y_pred=model.predict(X_test)
y_pred




model.score(X_test, y_test)




#step 4: Evaluate model:
cm = confusion_matrix(y_test,y_pred)
print (cm)




#step5: f-score :
print(f1_score(y_test, y_pred))




print(accuracy_score(y_test,y_pred))

