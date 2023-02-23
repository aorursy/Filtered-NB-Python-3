#!/usr/bin/env python
# coding: utf-8




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import sklearn as sk
import sklearn.preprocessing as prep
from sklearn import cross_validation

import numpy as np




import string
def deck(s):
    s = str(s)
    res = ''.join([c for c in s if c in string.ascii_uppercase])
    return res[0] if res else 'N'
def labels(series):
    enc = prep.LabelEncoder()
    enc.fit(series)
    return np.matrix(sexenc.transform(X['Sex'])).transpose()
def preprocess(X):
    y = X.copy()
    X = X[['Pclass','Sex', 'Cabin', 'Age','SibSp','Parch']]
    cabin = X['Cabin'].apply(deck)
    X['Cabin'] = cabin
    
    sex = labels(X['Sex'])
    print(X)
    cabin = X['Cabin']
    print(X)
    

train = pd.read_csv('../input/train.csv')
X= preprocess(train)
print(X)
y = train['Survived'].as_matrix()




f = RandomForestClassifier(max_features = 2, criterion = 'gini', n_estimators = 80)
scores = cross_validation.cross_val_score(f, X, y, cv=10, n_jobs=-1)
print(scores.mean(), scores.std())






classifiers = [LogisticRegression(C=1.0, penalty='l2', tol=1e-6),
               RandomForestClassifier(n_estimators=10, max_depth=5,
                                      min_samples_split=1, random_state=0),
               SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
              ExtraTreesClassifier()]
for c in classifiers:
    c.fit(X,y)
    scores = cross_validation.cross_val_score(c, X, y, cv=5, n_jobs=-1)
    print(c, scores.mean(), scores.std())




from sklearn.grid_search import GridSearchCV
   

forest = RandomForestClassifier()
params = {'criterion':['gini', 'entropy'], 'n_estimators':[10,20,40,80,160,400],'max_features':[1,2,3,4]}
searcher = GridSearchCV(estimator=forest, param_grid=params)
print(X)
searcher.fit(X,y)
searcher.score(X,y)
print(searcher.grid_scores_)




print(searcher.grid_scores_)




f = RandomForestClassifier(max_features = 2, criterion = 'gini', n_estimators = 80)
scores = cross_validation.cross_val_score(f, X, y, cv=10, n_jobs=-1)
print(scores.mean(), scores.std())

