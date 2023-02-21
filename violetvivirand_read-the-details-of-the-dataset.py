#!/usr/bin/env python
# coding: utf-8



import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib  # data visualization
import matplotlib.pyplot as plt  # data visualization shortcut
import seaborn as sns  # data visualization
from scipy import stats, integrate  # statistic
from sklearn.preprocessing import Imputer  # data imputation tool
                                           # http://scikit-learn.org/stable/modules/preprocessing.html#imputation

matplotlib.style.use('ggplot')  # use ggplot style
sns.set(color_codes=True)




# Load files
train_data = pd.read_csv('../input/train.csv')
train_data




train_data.isnull().sum()
# train_data.isnull().sum().plot.hist()




# Define age_data
age_data = train_data['Age']




# Read the info of age_data
age_data.dropna().describe()




# Calculate the Mode Value of Age Data
age_data.mode()




# View the distribution after categorized by [0,15,25,65,max_value]
## Use pandas.get_dummies() and pandas.cut() then calculate myself
age_bins = [0, 15, 25, 65, age_data.max()]
age_categoried = pd.get_dummies(pd.cut(age_data.dropna(), age_bins)).sum(axis=0)
age_categoried




(age_categoried/age_data.size).map('{:,.2%}'.format)




## Another way: just use pandas.cut() and use Seaborn to produce the plot
age_categoried_labels = list(map(lambda x: "["+ str(age_bins[x]) + " ~ " + str(age_bins[x+1]) + ")", [x for x in range(0,len(age_bins)-1)]))
age_cut = pd.DataFrame(pd.cut(age_data.dropna(), age_bins, labels=age_categoried_labels), columns=["Age"])
sns.countplot(x="Age", data=age_cut)




# View the original distributaion of age data
sns.distplot(age_data.dropna(), kde=False)




# View the Count Plot of original Age Data
f, axis1 = plt.subplots(1,1,figsize=(16,3))
sns.countplot(x='Age', data=train_data[['Age']].dropna().round().astype(np.int64), ax=axis1)




# Reshape the Age data so we can put it into imputer()
age_data = train_data['Age'].values.reshape(1,-1)  # If not reshaped,
                                                   # it shows feature deprecated message
                                                   # try to remove "reshape" function
                                                   # to read the information ;)




# Produce 3 Subplots
f,(axis0,axis1,axis2) = plt.subplots(1,3,figsize=(9,3))

# Mean = 29.699118
imp_mean = Imputer(missing_values='NaN', strategy='mean', axis=1)  # set axis=1
                                                                   # to impute along rows
imp_mean.fit(age_data)
## TODO:
## Use np.rint() to convert float to integer
## but still cast VisibleDeprecationWarning,
## check next stable version and #953: https://github.com/mwaskom/seaborn/issues/953
age_data_imputed_by_mean = np.rint(imp_mean.transform(age_data))   
sns.distplot(age_data_imputed_by_mean, hist=False, axlabel='Age of Passengers\r\n(Imputed by Mean)', ax=axis0)

# Median = 28
imp_median = Imputer(missing_values='NaN', strategy='median', axis=1)  # set axis=1
                                                                       # to impute along rows
imp_median.fit(age_data)
## TODO:
## Use np.rint() to convert float to integer
## but still cast VisibleDeprecationWarning,
## check next stable version and #953: https://github.com/mwaskom/seaborn/issues/953
age_data_imputed_by_median = np.rint(imp_median.transform(age_data))
sns.distplot(age_data_imputed_by_median, hist=False, axlabel='Age of Passengers\r\n(Imputed by Median)', ax=axis1)

# Mode = 24
imp_most_frequent = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)  # set axis=1
                                                                   # to impute along rows
imp_most_frequent.fit(age_data)
## TODO:
## Use np.rint() to convert float to integer
## but still cast VisibleDeprecationWarning,
## check next stable version and #953: https://github.com/mwaskom/seaborn/issues/953
age_data_imputed_by_most_frequent = np.rint(imp_most_frequent.transform(age_data))
sns.distplot(age_data_imputed_by_most_frequent, hist=False, axlabel='Age of Passengers\r\n(Imputed by Mode)', ax=axis2)

# Choose to Impute with Median Value
f, axis1 = plt.subplots(1,1,figsize=(16,3))
sns.countplot(x='Age', data=pd.DataFrame({'Age':age_data_imputed_by_most_frequent[0]}, dtype=np.int64), ax=axis1)

## 然後我發現這大有問題了
## 機率密度函數：年齡有小於零的......看來要再去了解機率密度函數的定義
##（還是繪圖的誤差？畢竟是把 Barchart 轉過來的？）
## 次數圖：遺漏值太多，看不出什麼鬼 QQ




# View the Distribution of Gender Data
## Value Counts
train_data['Sex'].value_counts()




## Gender and Survival Number
sex_survived_grouped = train_data.groupby(['Sex', 'Survived'])
for name, group in sex_survived_grouped:
    print('Gender: ' + name[0].title() + ',\t' +          'Survived: ' + ('No' if (name[1] == 0) else 'Yes' )+ ',\t' +          'Number of People: ' + str(group.shape[0]))




## Gender and Survival Ratio
sex_grouped = train_data.groupby(['Sex'])
sex_grouped.mean()['Survived'].map('{:,.2%}'.format)




## Count Plot devided by different Gender
plt.figure(figsize=(8,3))
sns.countplot(x="Sex", data=train_data)

## Count Plot devided by Different Gender and whether they Survived or not
f,(axis1,axis2) = plt.subplots(1,2,figsize=(8,3))

sns.countplot(x="Sex", hue='Survived', 
              data=train_data.replace({'Survived': {0: 'Not Survived', 1: 'Survived'}}),
              ax=axis1)
sns.countplot(x="Survived", hue='Sex', 
              data=train_data.replace({'Survived': {0: 'Not Survived', 1: 'Survived'}}),
              ax=axis2)

## Bar Plot: Show Different Gender and the Mean value of Survived
plt.figure(figsize=(8,3))
sns.barplot(x="Sex", y='Survived', data=train_data)




# View the Distribution of Class of Passengers Data
## Value Counts
train_data['Pclass'].value_counts().sort_index()




## Class of Passengers and Survival Number
pclass_survived_grouped = train_data.groupby(['Pclass', 'Survived'])
for name, group in pclass_survived_grouped:
    print('Class of Passenger: ' + str(name[0]) + ',\t' +          'Survived: ' + ('No' if (name[1] == 0) else 'Yes' )+ ',\t' +          'Number of People: ' + str(group.shape[0]))




## Gender and Survival Ratio
pclass_grouped = train_data.groupby(['Pclass'])
pclass_grouped.mean()['Survived'].map('{:,.2%}'.format).sort_index()




## Count Plot devided by different Class of Passengers
plt.figure(figsize=(8,3))
sns.countplot(x="Pclass", data=train_data)

## Count Plot devided by Different Class of Passengers and whether they Survived or not
f,(axis1,axis2) = plt.subplots(1,2,figsize=(8,3))

sns.countplot(x="Pclass", hue='Survived', 
              data=train_data.replace({'Survived': {0: 'Not Survived', 1: 'Survived'}}),
              ax=axis1)
sns.countplot(x="Survived", hue='Pclass', 
              data=train_data.replace({'Survived': {0: 'Not Survived', 1: 'Survived'}}),
              ax=axis2, order=['Survived', 'Not Survived'])

## Bar Plot: Show Different Gender and the Mean value of Survived
plt.figure(figsize=(8,3))
sns.barplot(x="Sex", y='Survived', data=train_data)




# Define Fare Data
fare_data = train_data['Fare']




# View the Description of Fare Data
fare_data.describe()




fare_data




# View the original distributaion of Fare data
sns.distplot(fare_data)




# View the distribution after categorized 
# by [0, First Quartile, Median, Third Quartile, Max]
## Use pandas.get_dummies() and pandas.cut() then calculate myself
fare_bins = [0, fare_data.quantile(0.25), fare_data.median(),             fare_data.quantile(0.75), fare_data.max()]
fare_categoried = pd.get_dummies(pd.cut(fare_data, fare_bins)).sum(axis=0).astype(np.int64)
fare_categoried




## Fare Count Ratio
(fare_categoried / fare_categoried.sum()).map('{:,.2%}'.format)




## Another way: just use pandas.cut() and use Seaborn to produce the plot
fare_categoried_labels = list(map(lambda x: "["+ str(fare_bins[x]) + " ~ " + str(fare_bins[x+1]) + ")", [x for x in range(0,len(fare_bins)-1)]))
fare_cut = pd.DataFrame(pd.cut(fare_data, fare_bins, labels=fare_categoried_labels), columns=["Fare"])
sns.countplot(x="Fare", data=fare_cut)




## Fare and Survival Number
categorized_data_survived_data = pd.DataFrame({'Fare': pd.cut(fare_data, fare_bins, labels=fare_categoried_labels),
                                               'Survived' : train_data['Survived']})
fare_survived_grouped = categorized_data_survived_data.groupby(['Fare', 'Survived'])
for name, group in fare_survived_grouped:
    print('Fare: ' + name[0] + ',\t' +          'Survived: ' + ('No' if (name[1] == 0) else 'Yes' )+ ',\t' +          'Number of People: ' + str(group.shape[0]))




## Fare and Survival Ratio
fare_grouped = categorized_data_survived_data.groupby(['Fare'])
fare_grouped.mean()['Survived'].map('{:,.2%}'.format)




## Fare and Survival Barplot
sns.barplot(x="Fare", y='Survived', data=categorized_data_survived_data)




# Define Embarked Data
embarked_data = train_data.replace({'Embarked': {'C' : 'Cherbourg',
                                                 'Q' : 'Queenstown',
                                                 'S' : 'Southampton'}})['Embarked']




# View the Distribution of Embarked Data
## Value Counts
embarked_data.value_counts().sort_index()




## Ratio of Each Embarked Place
(embarked_data.value_counts() / embarked_data.value_counts().sum()).sort_index().map('{:,.2%}'.format)




## Embarked and Survival Number
embarked_survived_grouped = train_data.replace({'Embarked': {'C' : 'Cherbourg',
                                                             'Q' : 'Queenstown',
                                                             'S' : 'Southampton'}})\
                                      .groupby(['Embarked', 'Survived'])
                                      
for name, group in embarked_survived_grouped:
    print('Embarked Place: ' + name[0] + ',\t' +          'Survived: ' + ('No' if (name[1] == 0) else 'Yes' )+ ',\t' +          'Number of People: ' + str(group.shape[0]))




## Embarked and Survival Ratio
embarked_grouped = train_data.replace({'Embarked': {'C' : 'Cherbourg',
                                                    'Q' : 'Queenstown',
                                                    'S' : 'Southampton'}})\
                             .groupby(['Embarked'])

embarked_grouped.mean()['Survived'].map('{:,.2%}'.format).sort_index()




## Count Plot devided by different Embarked Place
plt.figure(figsize=(8,3))
sns.countplot(x="Embarked", data=train_data.replace({'Embarked': {'C' : 'Cherbourg',
                                                                  'Q' : 'Queenstown',
                                                                  'S' : 'Southampton'}}),
              order=['Cherbourg', 'Queenstown', 'Southampton'])

## Count Plot devided by Different Gender and whether they Survived or not
f,(axis1,axis2) = plt.subplots(1,2,figsize=(8,3))

sns.countplot(x="Embarked", hue='Survived', 
              data=train_data.replace({'Survived': {0: 'Not Survived', 1: 'Survived'},
                                       'Embarked': {'C' : 'Cherbourg',
                                                    'Q' : 'Queenstown',
                                                    'S' : 'Southampton'}}),
              ax=axis1, order=['Cherbourg', 'Queenstown', 'Southampton'])
sns.countplot(x="Survived", hue='Embarked', 
              data=train_data.replace({'Survived': {0: 'Not Survived', 1: 'Survived'},
                                       'Embarked': {'C' : 'Cherbourg',
                                                    'Q' : 'Queenstown',
                                                    'S' : 'Southampton'}}),
              ax=axis2, order=['Survived', 'Not Survived'])

## Bar Plot: Show Different Gender and the Mean value of Survived
plt.figure(figsize=(8,3))
sns.barplot(x="Embarked", y='Survived',
            data=train_data.replace({'Embarked': {'C' : 'Cherbourg',
                                                  'Q' : 'Queenstown',
                                                  'S' : 'Southampton'}}),
            order=['Cherbourg', 'Queenstown', 'Southampton'])




## Regression

