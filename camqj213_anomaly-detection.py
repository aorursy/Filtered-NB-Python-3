#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import random

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




data = pd.read_csv("../input/creditcard.csv")
data.head()
# Overview of the input data




data.isnull().sum()




fraudAmount = set(data[data.Class == 1].Amount)




corr = data.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)




X = np.array(data)
np.random.shuffle(X)

y = X[:,30]
X = X[:,0:30]
print(y)
print(X)




standard_scaler = StandardScaler()
scaled_X = standard_scaler.fit_transform(X)
print(scaled_X)




n_samples = len(scaled_X)
partition = int(.9 * n_samples)
X_train = scaled_X[:partition]
y_train = y[:partition]
X_test = scaled_X[partition:]
y_test = y[partition:]




logistic = linear_model.LogisticRegression()
print('LogisticRegression score: %f'
      % logistic.fit(X_train, y_train).score(X_test, y_test))




predicted = logistic.predict(X_test)
print(predicted)
print(len(predicted), len(y_test))
true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
for p, i in zip(predicted, y_test):
    if p == 0 and i == 0:
        true_negative += 1
    elif p == 0 and i == 1:
        false_negative += 1
    elif p == 1 and i == 1:
        true_positive += 1
    elif p == 1 and i == 0:
        false_positive += 1
print(true_positive, false_positive)
print(false_negative, true_negative)

