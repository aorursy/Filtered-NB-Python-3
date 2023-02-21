#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from subprocess import check_output
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
del train['PassengerId']
del train['Name']
del train['Ticket']
del train['Cabin']
train = train.dropna()
t = train[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
le = LabelEncoder()
t[:, 2] = le.fit_transform(t[:, 2])
t[:, 7] = le.fit_transform(t[:, 7])
ohe = OneHotEncoder(categorical_features=[7])
ohe.fit_transform(t).toarray()
train = pd.get_dummies(train[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]])
train_x = train.ix[:, 1:]
train_y = train.ix[:, 0]
feat_labels = train_x.columns[:]




forest2 = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest2.fit(train_x, train_y)
importances = forest2.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(train_x.shape[1]):
    print(feat_labels[indices[f]], importances[indices[f]])




plt.clf()
plt.title("Feature Importance")
plt.bar(range(train_x.shape[1]), importances[indices], color="green", align="center")
plt.xticks(range(train_x.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, train_x.shape[1]])
plt.tight_layout()
plt.show




train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)




train[['Sex_female', 'Survived']].groupby(['Sex_female'], as_index=False).mean().sort_values(by='Survived', ascending=False)




train_beta = np.array(train[["Age", "Survived"]])
for i in range(len(train_beta)):
    if train_beta[i, 0] < 20:
        train_beta[i, 0] = 0
    elif train_beta[i, 0] < 30:
        train_beta[i, 0] = 1
    elif train_beta[i, 0] < 40:
        train_beta[i, 0] = 2
    elif train_beta[i, 0] < 50:
        train_beta[i, 0] = 3
    else:
        train_beta[i, 0] = 4
train_beta = pd.DataFrame({'Age': train_beta[:, 0], "Survived": train_beta[:, 1]})
train_beta[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)




train_beta = np.array(train[["Fare", "Survived"]])
for i in range(len(train_beta)):
    if train_beta[i, 0] < 10:
        train_beta[i, 0] = 0
    elif train_beta[i, 0] < 50:
        train_beta[i, 0] = 1
    else:
        train_beta[i, 0] = 2
train_beta = pd.DataFrame({"Fare": train_beta[:, 0], "Survived": train_beta[:, 1]})
train_beta[["Fare", 'Survived']].groupby(["Fare"], as_index=False).mean().sort_values(by='Survived', ascending=False)




del train_x['Sex_male']




sc = StandardScaler()
tr_x, te_x, tr_y, te_y = train_test_split(train_x, train_y, test_size=0.15, random_state=0)
tr_x_std = sc.fit_transform(tr_x)
te_x_std = sc.transform(te_x)




def Logistic(i, tr_x_std, tr_y, te_x_std, te_y):
    lr = LogisticRegression()
    lr.fit(tr_x_std, tr_y)
    y1 = lr.predict(tr_x_std)
    y2 = lr.predict(te_x_std)
    acc_tr = accuracy_score(y1, tr_y)
    acc_te = accuracy_score(y2, te_y)
    return acc_tr, acc_te
def SVM(i, tr_x_std, tr_y, te_x_std, te_y):
    svm = SVC()
    svm.fit(tr_x_std, tr_y)
    y1 = svm.predict(tr_x_std)
    y2 = svm.predict(te_x_std)
    acc_tr = accuracy_score(y1, tr_y)
    acc_te = accuracy_score(y2, te_y)
    return acc_tr, acc_te
def kerbel_SVM(i, tr_x_std, tr_y, te_x_std, te_y):
    svm = SVC(kernel="rbf")
    svm.fit(tr_x_std, tr_y)
    y1 = svm.predict(tr_x_std)
    y2 = svm.predict(te_x_std)
    acc_tr = accuracy_score(y1, tr_y)
    acc_te = accuracy_score(y2, te_y)
    return acc_tr, acc_te
def randomforest(i, tr_x, tr_y, te_x, te_y):
    forest = RandomForestClassifier()
    forest.fit(tr_x, tr_y)
    y1 = forest.predict(tr_x)
    y2 = forest.predict(te_x)
    acc_tr = accuracy_score(y1, tr_y)
    acc_te = accuracy_score(y2, te_y)
    return acc_tr, acc_te




feature = ["Fare", "Sex_female", "Pclass", "SibSp", "Parch", "Embarked_C", "Embarked_S", "Embarked_Q"]
xdata_tr = pd.DataFrame(tr_x["Age"])
xdata_te = pd.DataFrame(te_x["Age"])
acc_logistic_tr = []
acc_logistic_te = []
acc_svm_tr = []
acc_svm_te = []
acc_kernel_tr = []
acc_kernel_te = []
acc_random_tr = []
acc_random_te = []
log_a = Logistic(i, xdata_tr, tr_y, xdata_te, te_y)
acc_logistic_tr.append(log_a[0])
acc_logistic_te.append(log_a[1])
svm_a = SVM(i, xdata_tr, tr_y, xdata_te, te_y)
acc_svm_tr.append(svm_a[0])
acc_svm_te.append(svm_a[1])
ker_a = kerbel_SVM(i, xdata_tr, tr_y, xdata_te, te_y)
acc_kernel_tr.append(ker_a[0])
acc_kernel_te.append(ker_a[1])
rand_a = randomforest(i, xdata_tr, tr_y, xdata_te, te_y)
acc_random_tr.append(rand_a[0])
acc_random_te.append(rand_a[1])
for i, f in enumerate(feature):
    print(i+1, f)
    xdata_tr[f] = tr_x[f]
    xdata_te[f] = te_x[f]
    log_a = Logistic(i, xdata_tr, tr_y, xdata_te, te_y)
    acc_logistic_tr.append(log_a[0])
    acc_logistic_te.append(log_a[1])
    svm_a = SVM(i, xdata_tr, tr_y, xdata_te, te_y)
    acc_svm_tr.append(svm_a[0])
    acc_svm_te.append(svm_a[1])
    ker_a = kerbel_SVM(i, xdata_tr, tr_y, xdata_te, te_y)
    acc_kernel_tr.append(ker_a[0])
    acc_kernel_te.append(ker_a[1])
    rand_a = randomforest(i, xdata_tr, tr_y, xdata_te, te_y)
    acc_random_tr.append(rand_a[0])
    acc_random_te.append(rand_a[1])




plt.clf()
plt.title("Feature Logistic")
plt.plot(range(9), acc_logistic_tr, c="green", label="train")
plt.plot(range(9), acc_logistic_te, c="red", label="test")
plt.legend(loc="upper left")
plt.show




plt.clf()
plt.title("Feature SVM")
plt.plot(range(9), acc_svm_tr, c="green", label="train")
plt.plot(range(9), acc_svm_te, c="red", label="test")
plt.legend(loc="upper left")
plt.show




plt.clf()
plt.title("Feature Kernel")
plt.plot(range(9), acc_kernel_tr, c="green", label="train")
plt.plot(range(9), acc_kernel_te, c="red", label="test")
plt.legend(loc="upper left")
plt.show




plt.clf()
plt.title("Feature Randomforest")
plt.plot(range(9), acc_random_tr, c="green", label="train")
plt.plot(range(9), acc_random_te, c="red", label="test")
plt.legend(loc="upper left")
plt.show




del train_x["SibSp"]
del train_x["Parch"]
del train_x["Embarked_C"]
del train_x["Embarked_S"]
del train_x["Embarked_Q"]
sc = StandardScaler()
tr_x, te_x, tr_y, te_y = train_test_split(train_x, train_y, test_size=0.15, random_state=0)
tr_x_std = sc.fit_transform(tr_x)
te_x_std = sc.transform(te_x)




lr_tr = []
lr_te = []
params = []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(tr_x_std, tr_y)
    y1 = lr.predict(tr_x_std)
    y2 = lr.predict(te_x_std)
    acc_tr = accuracy_score(y1, tr_y)
    acc_te = accuracy_score(y2, te_y)
    lr_tr.append(acc_tr)
    lr_te.append(acc_te)
    params.append(c)
plt.clf()
plt.title("Logistic Regression")
plt.plot(params, lr_tr, c="green", label="train")
plt.plot(params, lr_te, c="red", label="test")
plt.legend(loc="upper left")
plt.show




forest_tr = []
forest_te = []
params = []
count = 0
for n in range(1, 10):
    forest = RandomForestClassifier(n_estimators=n, random_state=0, n_jobs=-1)
    forest.fit(tr_x, tr_y)
    y1 = forest.predict(tr_x)
    y2 = forest.predict(te_x)
    acc_tr = accuracy_score(y1, tr_y)
    acc_te = accuracy_score(y2, te_y)
    forest_tr.append(acc_tr)
    forest_te.append(acc_te)
    params.append(n)
plt.clf()
plt.title("Random Forest")
plt.plot(params, forest_tr, c="green", label="train")
plt.plot(params, forest_te, c="red", label="test")
plt.legend(loc="upper left")
plt.show






