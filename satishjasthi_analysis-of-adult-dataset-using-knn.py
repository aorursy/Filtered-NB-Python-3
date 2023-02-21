#!/usr/bin/env python
# coding: utf-8



#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split,KFold,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')




#loading dataset
df = pd.read_csv('../input/adult.csv')
df.head()




#number of features
print ("Number of features : {}".format(len(df.columns.values)))
print ("Number of sample : {}".format(df.shape[0]))




#let see if any cloumn has missing values
df.info()




print('workclass\n',set(df.workclass))
print('\neducation\n',set(df.education))
print('\nmarital.status\n',set(df['marital.status']))
print('\noccupation\n',set(df.occupation))
print('\nrace\n',set(df.race))
print('\nrelationship',set(df.relationship))
print('\nsex',set(df.sex))
print('\nnative.country',set(df['native.country']))
print('\nsalary',set(df.income))




set(df['income'].values)




df.workclass = df.workclass.map({ '?':0, 'Federal-gov':1, 'Local-gov':2, 'Never-worked':3, 'Private':4, 'Self-emp-inc':5, 'Self-emp-not-inc':6, 'State-gov':7, 'Without-pay':8})

df.income = np.where(df.income == '>50K',1,0)

df.occupation = df.occupation.map({'?':0, 'Adm-clerical':1, 'Armed-Forces':2, 'Craft-repair':3, 'Exec-managerial':4, 'Farming-fishing':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Other-service':8,'Priv-house-serv':9,'Prof-specialty':10,'Protective-serv':11,'Sales':12,'Tech-support':13,'Transport-moving':14})

df['marital.status'] = df['marital.status'].map({'Divorced':0,'Married-AF-spouse':1,'Married-civ-spouse':2,'Married-spouse-absent':3,'Never-married':4,'Separated':5,'Widowed':6})

df.race = df.race.map({'Amer-Indian-Eskimo':0, 'Asian-Pac-Islander':1, 'Black':2, 'Other':3, 'White':4})

df.sex = np.where(df.sex == 'Male',1,0)

df.relationship = df.relationship.map({'Husband':0,'Not-in-family':1,'Other-relative':2,'Own-child':3,'Unmarried':4,'Wife':5})

df['native.country'] = df['native.country'].map({'?':0,'Cambodia':1,'Canada':2,'China':3,'Columbia':4,'Cuba':5,'Dominican-Republic':6,'Ecuador':7,
 'El-Salvador':8,'England':9,'France':10,'Germany':11,'Greece':12,'Guatemala':13,'Haiti':14,'Holand-Netherlands':15,'Honduras':16,
 'Hong':17,'Hungary':18,'India':19,'Iran':20,'Ireland':21,'Italy':22,'Jamaica':23,'Japan':24,'Laos':25,'Mexico':26,'Nicaragua':27,
 'Outlying-US(Guam-USVI-etc)':28,'Peru':29,'Philippines':30,'Poland':31,'Portugal':32,'Puerto-Rico':33,'Scotland':34,
 'South':35,'Taiwan':36,'Thailand':37,'Trinadad&Tobago':38,'United-States':39,'Vietnam':40,'Yugoslavia':4})

df.education = df.education.map({'10th':0,'11th':1,'12th':2,'1st-4th':3,'5th-6th':4,'7th-8th':5,'9th':6,'Assoc-acdm':7,'Assoc-voc':8,
 'Bachelors':9,'Doctorate':10,'HS-grad':11,'Masters':12,'Preschool':13,'Prof-school':14,'Some-college':15})

df.head()




features = df.drop('income',axis=1)
target = df.income




#define a classifier
model = LogisticRegression()

#create RFE model to return top 3 attributes
rfe = RFE(model,3)
rfe = rfe.fit(features,target)

#summarise the selection of attributes
print('\n rfe.support:\n',rfe.support_)
print('\n rfe.ranking:\n',rfe.ranking_)
print('\n features:\n',features.columns.values)




#define and fit a ExtraTreeClassifier to the data
model = ExtraTreesClassifier()
model.fit(features,target)

#display the feature importance
print(model.feature_importances_)
print('\n',features.columns.values)




#bar plot of feature importance
values = model.feature_importances_
pos = np.arange(14) + 0.02
plt.barh(pos,values,align = 'center')
plt.title('Feature importance plot')
plt.xlabel('feature importance ')
plt.ylabel('features')
plt.yticks(np.arange(14),('age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status','occupation' ,'relationship', 'race' ,'sex', 'capital-gain', 'capital.loss','hours.per.week', 'native.country'))
plt.grid(True)




#updating features: combining best features from both RFE and feature importance
features = features[['education.num','marital.status','relationship','race','age','fnlwgt']]

#here we have consider best features from both RFE and feature importance results

#spliting data into train and test data
X_train,X_test,y_train,y_test = train_test_split(features,target,random_state = 12)

from sklearn.neighbors import KNeighborsClassifier

k_values = np.arange(1,26)
scores = []

for i in k_values:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_predict))

print("Accuracy for {} is {}".format(np.argmax(scores),max(scores)))

plt.plot(np.arange(1,26),scores)
plt.title('Varition of accuracy with K value,with best features from RFE and feature importance')
plt.xlabel('K values')
plt.ylabel('Accuracy')




#Let's update features with the results of RFE and evaluate how
#accuracy varies

features1 = features[['education.num','marital.status','relationship','race']]

X_train,X_test,y_train,y_test = train_test_split(features1,target,random_state = 12)

from sklearn.neighbors import KNeighborsClassifier

k_values = np.arange(1,26)
scores = []

for i in k_values:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_predict))

print("Accuracy for {} is {}".format(np.argmax(scores),max(scores)))

plt.plot(np.arange(1,26),scores)
plt.title('Varition of accuracy with K value,with best features from RFE ')
plt.xlabel('K values')
plt.ylabel('Accuracy')




#Let's update features with the results of feature importance and evaluate how
#accuracy varies

features2 = features[['age','fnlwgt']]

X_train,X_test,y_train,y_test = train_test_split(features2,target,random_state = 12)

from sklearn.neighbors import KNeighborsClassifier

k_values = np.arange(1,26)
scores = []

for i in k_values:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_predict))

print("Accuracy for {} is {}".format(np.argmax(scores),max(scores)))

plt.plot(np.arange(1,26),scores)
plt.title('Varition of accuracy with K value,with best features from feature importance')
plt.xlabel('K values')
plt.ylabel('Accuracy')




from sklearn.svm import SVC
c = np.arange(0.1,1.1,0.1)
scores = {}
for value in c:
    clf = SVC(C = value,kernel='linear')
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    scores.setdefault(value,metrics.accuracy_score(y_test,predictions))
print scores




np.arange(0,1.1,0.1)






