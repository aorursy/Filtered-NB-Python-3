#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import random as rnd

import os
# for data visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier





#work directory
# os.getcwd




# csv to data fi
df=pd.read_csv("../input/train.csv")




df.head(10)




df.tail(10)




df.info()




# data can be manupulated 
df['Age'][0:15]




df.Age[15:30]




df[['Age','Survived','Embarked']][0:15]




# data filtering
df[df.Age>45]




df[df.Age<45]




# We should show all the cases with null Age
df[df.Age.isnull()]




# it is important that we clean the data at this stage
df['Gender']=1 #Adding one colomun 




df['Gender']=df['Sex'].map({'female':0, 'male':1}).astype(int)
df.head(15)




df['Gender']=df['Sex'].map({'female':0, 'male':1}).astype(int)
df.head(30)




df['Gender']=df['Sex'].map({'female':0, 'male':1}).astype(int)
df.tail(15)




df.shape # number of rows and columns




df_duplicated=df.drop_duplicates()  #removing duplicated
df.shape




df.shape




df.rename(index={"NaN": 0})




df.shape




#Discretization and binning
df_bin=df.iloc[:10 :1]
df_bin




# drop any missing data
df.dropna()




df['Age'].hist()




# we can observe significant correlation between Survived and Pclass
df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)




# The variable Sex tells us that women have higher surviving rate than their counterpart men.
df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)




# some of the values in SibSp has zero correlations 
df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)




# similar to Sibsp we have zerp values in our data...so we decide to include them in our model
df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)




df.describe()
# through visualization we observ




d = sns.FacetGrid(df, col='Survived')
d.map(plt.hist, 'Age', bins=20)

