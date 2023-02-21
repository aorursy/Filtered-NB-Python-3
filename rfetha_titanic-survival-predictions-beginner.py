#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




sns.barplot(x="Sex" , y="Survived" , data=train)

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(
        normalize = True)[1]*100)

print("Percentage of females who survived:" , train["Survived"][train["Sex"] == 'male'].value_counts(
        normalize = True)[1]*100)




sns.barplot(x="SibSp" , y="Survived" , data=train)

print("Percentage of SibSp = 0 who survived:" , train["Survived"][train["SibSp"] == 0].value_counts(
            normalize = True )[1]*100 )

print("Percentage of SibSp = 1 who survived:" , train["Survived"][train["SibSp"] == 0].value_counts(
            normalize = True )[1]*100 )

print("Percentage of SibSp = 2 who survived:" , train["Survived"][train["SibSp"] == 0].value_counts(
            normalize = True )[1]*100 )




sns.barplot(x="Parch", y="Survived", data=train)
plt.show()




train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()




train["CabinBool"] = (train["Cabin"].notnull().astype("int"))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

print("Percentage of CabinBool = 0 who survived:" , train["Survived"][train["CabinBool"] == 0].value_counts(
            normalize = True )[1]*100 )

print("Percentage of CabinBool = 1 who survived:" , train["Survived"][train["CabinBool"] == 0].value_counts(
            normalize = True )[1]*100 )

print("Percentage of CabinBool = 2 who survived:" , train["Survived"][train["CabinBool"] == 0].value_counts(
            normalize = True )[1]*100 )






