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
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.head())
print(test.head())




print(train.describe())




print(train.isnull().sum())




print(train.nunique())




print(train.corr())
train.corr()['Survived'].sort_values()
#From this it looks like Pclass, Fare, and Sex correlate the most with Survival so let's focus
#on those features first




print(set(train.Sex))
print(set(test.Sex))
train.Sex = train.Sex.map(lambda s: 1 if s == 'female' else 0)
test.Sex = test.Sex.map(lambda s: 1 if s == 'female' else 0)
print(set(train.Sex))
print(set(test.Sex))




print('Min Age: %f, Max Age: %f, Median Age: %f, Mean Age: %f'
         % (train.Age.min(), train.Age.max(), train.Age.median(), train.Age.mean()))
test['Age'].fillna(test.Age.median(), inplace=True)
train['Age'].fillna(train.Age.median(), inplace=True) 
#Need to find some bins for ages




train['famshere'] = train.apply(lambda r: 1 if r['SibSp'] !=0 or r['Parch'] !=0 else 0, axis=1)
test['famshere'] = test.apply(lambda r: 1 if r['SibSp'] !=0 or r['Parch'] !=0 else 0, axis=1)




del train['SibSp'], train['Parch']
del test['SibSp'], test['Parch']




x = train[['Pclass', 'Sex', 'Age', 'famshere']].copy().values
y = train['Survived'].values
test_x = test[['Pclass', 'Sex', 'Age', 'famshere']].copy().values

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(x,y)
test_y = clf.predict(test_x)
test['Survived'] = test_y




print(test.head())
print(len(test))
test[['PassengerId', 'Survived']].to_csv('1st_tree_try.csv', index=False)

