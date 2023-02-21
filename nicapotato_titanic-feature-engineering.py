#!/usr/bin/env python
# coding: utf-8



# General
import numpy as np
import pandas as pd

# Normalizer
from sklearn import preprocessing




# Read Data
train_df = pd.read_csv("../input/train.csv", index_col='PassengerId')
test_df = pd.read_csv("../input/test.csv", index_col='PassengerId')
Survived = train_df['Survived'].copy()
train_df = train_df.drop('Survived', axis=1)




test_df.shape, train_df.shape




# Combine Test and Train to perform feature engineering all at once
df = pd.concat([test_df, train_df])
traindex = train_df.index
testdex = test_df.index
print(test_df.equals(df.loc[testdex,:]))
print(train_df.equals(df.loc[traindex,:]))
del train_df
del test_df




df.head()




# Proportion Missing Table:
settypes=df.dtypes.reset_index()
def missing(df):
    missing = df.isnull().sum(axis=0).reset_index()
    missing.columns = ['column_name', 'missing_count']
    missing['missing_ratio'] = missing['missing_count'] / df.shape[0]
    missing = pd.merge(missing,settypes, left_on='column_name', right_on='index',how='inner')
    missing = missing.loc[(missing['missing_ratio']>0)]    .sort_values(by=["missing_ratio"], ascending=False)
    return missing




mis = missing(df)
mis




# New Variables engineering, heavily influenced by:
# Kaggle Source- https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Name Length
df['Name_length'] = df['Name'].apply(len)
# Is Alone?
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1




# Title: (Source)
# Kaggle Source- https://www.kaggle.com/ash316/eda-to-prediction-dietanic
df['Title']=0
df['Title']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col',
                         'Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)




df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= df.Age[df.Title=="Mr"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= df.Age[df.Title=="Mrs"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= df.Age[df.Title=="Master"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= df.Age[df.Title=="Miss"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= df.Age[df.Title=="Other"].mean()
df = df.drop('Name', axis=1)




# Fill NA
# Categoricals Variable
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
# Continuous Variable
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())




## Assign Binary to Sex str
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# Title
#df['Title'] = df['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Rare':4} ).astype(int)
# Embarked
df['Embarked'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

# Get Rid of Ticket and Cabin Variable
df= df.drop(['Ticket', 'Cabin'], axis=1)




# Scaling between -1 and 1. Good practice for continuous variables.
from sklearn import preprocessing
for col in ['Fare','Age','Name_length']:
    transf = df[col].reshape(-1,1)
    scaler = preprocessing.StandardScaler().fit(transf)
    df[col] = scaler.transform(transf)




train_df = df.loc[traindex, :]
train_df['Survived'] = Survived




train_df.to_csv('clean_train_nick.csv',header=True,index=True)
df.loc[testdex, :].to_csv('clean_test_nick.csv',header=True,index=True)




train_df.head()




df.describe()

