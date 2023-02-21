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




import csv as csv
csv_file_object = csv.reader(open('../input/train.csv', 'r'))
next(csv_file_object)
data = []

for row in csv_file_object:
    data.append(row)
data = np.array(data)




print(data)




data[0:15, 5]




type(data[0::, 5])




ages_onboard = data[0::,5].astype(np.float)




df = pd.read_csv('../input/train.csv', header=0)




df




df.head(3)




type(df)




df.dtypes




df.info()




df.describe()




df['Age'][0:10]




df.Age[0:10]




df.Cabin




type(df['Age'])




df['Age'].mean()




df['Age'].median()




df[ ['Sex', 'Pclass', 'Age'] ]




df[df['Age'] > 60]




df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]




df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]




for i in range(1, 4):
    print(i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))




import pylab as P
df['Age'].hist()




P.show()




df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)




P.show()




df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )




df.head()




df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)




df.head()




median_ages = np.zeros((2,3))




median_ages




for i in range(2):
    for j in range(3):
        median_ages[i, j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1) ]['Age'].dropna().median()




median_ages




df['AgeFill'] = df['Age']




df.head()




df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)




for i in range(2):
    for j in range(3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]




df[ df.Age.isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)




df['AgeIsNull'] = pd.isnull(df.Age).astype(int)




df.describe()




df['FamilySize'] = df['SibSp'] + df['Parch']




df['Age*Class'] = df.AgeFill * df.Pclass




df['Age*Class'].hist()




df.info()




df.dtypes




df.dtypes[df.dtypes.map(lambda x: x == 'object')]




df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)




df = df.drop(['Age'], axis=1)




train_data = df.values




train_data




data




df.head()




df = df.drop(['PassengerId'], axis=1)




train_data = df.values




from sklearn.ensemble import RandomForestClassifier




forest = RandomForestClassifier(n_estimators = 100)




forest = forest.fit(train_data[0::,1::],train_data[0::,0])




output = forest.predict(test_data)






