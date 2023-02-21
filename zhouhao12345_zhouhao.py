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




import pandas as pd
import numpy as np

df = pd.read_csv('../input/train.csv', header=0)




df




df.head(3)




type(df)




df.dtypes




df.info()




df.describe()




df['Age'][0:10]




type(df['Age'])




df['Age'].mean()




df[ ['Sex', 'Pclass', 'Age'] ]




df[df['Age'] > 60]




df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]




df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]




for i in range(1,4):
    print (i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ]))




import pylab as P
df['Age'].hist()
P.show()




df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()




df['Gender'] = 4




df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )




df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)




median_ages = np.zeros((2,3))
median_ages




for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) &                               (df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages




df['AgeFill'] = df['Age']

df.head()




df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)




for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),                'AgeFill'] = median_ages[i,j]




df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)




df['AgeIsNull'] = pd.isnull(df.Age).astype(int)




df['FamilySize'] = df['SibSp'] + df['Parch']





df['Age*Class'] = df.AgeFill * df.Pclass




df.dtypes




df.dtypes[df.dtypes.map(lambda x: x=='object')]




df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1) 




df = df.drop(['Age'], axis=1)




df = df.dropna()




df.info




df.info()




train_data = df.values
train_data




import matplotlib.pyplot as plt
fig=plt.figure()
fig.set(alpha=0.4)   #设定图表颜色参数
df.Survived.value_counts().plot(kind='bar')    #用柱状图统计幸存人数
plt.title(u'获救情况(1为获救)')#设标题
plt.ylabel(u'人数')    #y轴表示人数




plt.scatter(df.Survived,df.AgeFill)    #用散点图表示不同年龄的获救情况
plt.ylabel(u'年龄')
plt.title(u'按年龄看获救分布(1为获救)')

