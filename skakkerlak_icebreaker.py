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




train = pd.read_csv('../input/train.csv')




train.info()




test = pd.read_csv('../input/test.csv')




test.info()




train.head()




train['Survived'].value_counts(normalize=True)




train['Survived'].groupby(train['Sex']).mean()




train['Sex'].value_counts()




train['Sex'].groupby(train['Embarked']).value_counts()




train['Sex'][train['Survived']==1].groupby(train['Embarked']).value_counts()




train.groupby(['Sex','Embarked'])['Survived'].mean()




train.groupby(['Sex','Embarked'])['Fare'].mean()




train.groupby(['Sex','Embarked'])['Age'].mean()




train.groupby(['Sex','Embarked'])['Pclass'].mean()




train.groupby(['Sex','Embarked'])['Pclass'].value_counts()




train.groupby(['Sex','Embarked','Pclass'])['Survived'].mean()




import seaborn as sns





sns.jointplot(x='Age',y='Fare',data=train)




sns.jointplot(x='Age',y='Fare',data=train[train['Fare']<500])




sns.lmplot(x='Age',y='Fare',hue='Pclass',data=train[train['Fare']<500])




Not surprisingly class and fare are correlated. And so are age and class. Very few children below the age of 10 were in first class. It seems there was no special fare for children. 

The pattern however has some degree of complexity. A handfull passengers get to travel 1st class at a very cheap rate.




sns.jointplot(x='Age',y='Fare',data=train[train['Fare']<500][train['Pclass']==1])




firstclass=train[train['Pclass']==1]
sns.lmplot(x='Age',y='Fare',hue='Embarked',data=firstclass[firstclass['Fare']<500])

