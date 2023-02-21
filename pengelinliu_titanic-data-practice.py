#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd
import os
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




#Visulization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




df=pd.read_csv('../input/Titanic.csv')




df.head(5)
df.info()
df.columns.values




df.describe()




df.describe(include=['O'])




sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df);




df[['Pclass', 'Survived']].groupby(['Pclass'],     as_index=False).mean().sort_values(by='Survived', ascending=False)




sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=df,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);




g = sns.FacetGrid(df, col='Pclass')
g.map(plt.hist, 'Age', bins=20)




sns.pointplot(x="Pclass", y="Fare", hue="Sex", data=df,
              palette={"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"]);




#mapping data
df['Sex']=df['Sex'].map({'male':1, 'female':0}).astype(int)
df['Sex'].head(5)




age_avg 	   = df['Age'].mean()
age_std 	   = df['Age'].std()
age_null_count = df['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
df['Age'][np.isnan(df['Age'])] = age_null_random_list
df['Age'] = df['Age'].astype(int)
df['Age'].head(5)




df['CatergoryAge']=pd.cut(df['Age'],5) 
print (df[['CatergoryAge', 'Survived']].groupby(['CatergoryAge'], as_index=False).mean()) 




def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

df['Title'] = df['Name'].apply(get_title)

print(pd.crosstab(df['Title'], df['Sex']))

