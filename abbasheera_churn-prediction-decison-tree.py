#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go#visualization
import plotly.offline as py#visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#from sklearn import cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




df=pd.read_csv("../input/churn.csv")




print ("Rows     : " ,df.shape[0])
print ("Columns  : " ,df.shape[1])
print ("\nFeatures : \n" ,df.columns.tolist())
print ("\nMissing values :  ", df.isnull().sum().values.sum())
print ("\nUnique values :  \n",df.nunique())




df.shape




df.describe().T




df.columns




df.drop(columns={'State','Area Code', 'Phone'}, inplace=True)
df.columns




df = df.rename(columns={'Churn?': 'Churn'})




df['Churn'] = df['Churn'].map(lambda x:'Yes' if x=='True.' else 'No')




df.head()




lab = df["Churn"].value_counts().keys().tolist()
lab




val = df["Churn"].value_counts().values.tolist()
val




labels = df["Churn"].value_counts().keys().tolist()
sizes = df["Churn"].value_counts().values.tolist()
colors = ['YellowGreen', 'Red', ]
explode = (0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.2f%%', shadow=True, startangle=360)

plt.axis('equal')
plt.show()




df['Churn'] = df['Churn'].map(lambda x:1 if x=='Yes' else 0)
df["Int'l Plan"] = df["Int'l Plan"].map(lambda x:1 if x=='yes' else 0)
df["VMail Plan"] = df['VMail Plan'].map(lambda x:1 if x=='yes' else 0)




df.head()




# No Null Values & No Duplicate Values 
df.duplicated().sum()
df.isnull().sum()




df.dtypes




df.corr()["Churn"].sort_values()




df.groupby(["Int'l Plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 




df.groupby(["VMail Plan", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 




df.groupby(["CustServ Calls", "Churn"]).size().unstack().plot(kind='bar', stacked=True, figsize=(5,5)) 




X=df[["Int'l Plan", "VMail Plan","CustServ Calls"]]
X.head()
x=df[["Int'l Plan", "VMail Plan","CustServ Calls"]]





Y = df['Churn']
Y.head()





scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)




X




from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 




X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)




train_pred = logreg.predict(X_train)




test_pred = logreg.predict(X_test)




from sklearn import metrics
metrics.confusion_matrix(Y_train,train_pred)




print("Accuracy Score for Train using Logistic Regression :",metrics.accuracy_score(Y_train,train_pred))




print("Accuracy Score for Test using Logistic Regression :" ,metrics.accuracy_score(Y_test, test_pred))




from sklearn.metrics import classification_report
print(classification_report(Y_train, train_pred))




from sklearn.metrics import classification_report
print(classification_report(Y_test, test_pred))




#  decision Tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree




clf_gini = DecisionTreeClassifier(criterion = "gini",
                               max_depth=3, min_samples_leaf=5)
clf=clf_gini.fit(X_train, Y_train)




y_train_pred = clf_gini.predict(X_train)
print("Accuracy Score for Train Using Decision Tree Classifier : ", accuracy_score(Y_train, y_train_pred))




y_pred = clf_gini.predict(X_test)





#pip install graphviz









from IPython.display import Image 
from pandas import DataFrame, Series
#from cStringIO  import StringIO
from io import StringIO
import pydotplus




def plot_decision_tree(clf,feature_name,target_name):
    dot_data = StringIO()  
    tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=feature_name,  
                         class_names=target_name,  
                         filled=True, rounded=True,  
                         special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    return Image(graph.create_png())




#plot_decision_tree(clf, X_train.columns,df.columns[1])

from sklearn.tree import DecisionTreeClassifier,export_graphviz




feat_names = x.columns
targ_names =['Yes','No']




import graphviz




data = export_graphviz(clf,out_file=None,feature_names=feat_names,class_names=targ_names,   
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(data)
graph




x.columns

