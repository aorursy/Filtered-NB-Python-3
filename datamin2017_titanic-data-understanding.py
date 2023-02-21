#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.stats.stats import pearsonr




df = pd.read_csv("titanic_train.csv") 




df.head()




df.tail()




df.dtypes




df.info()




df.describe()




df[['Age', 'Survived']].head()




df.Age.head()









df.Pclass.corr(df['Age'])




df.corr()




plt.scatter(df['Age'], df['Fare'], color='g')




pd.scatter_matrix(df[['Age', 'Fare', 'Pclass', 'Parch']])




df['Age'].head()




df[['Age', 'Sex']].head()




# Set up a grid of plots
fig = plt.figure(figsize=(10, 10)) 
fig_dims = (3, 2)

# Plot death and survival counts
plt.subplot2grid(fig_dims, (0, 0))
df['Survived'].value_counts().plot(kind='bar', title='Death and Survival Counts')

# Plot Pclass counts
plt.subplot2grid(fig_dims, (0, 1))
df['Pclass'].value_counts().plot(kind='bar', title='Passenger Class Counts')

# Plot Sex counts
plt.subplot2grid(fig_dims, (1, 0))
df['Sex'].value_counts().plot(kind='bar', title='Gender Counts')
plt.xticks(rotation=0)

# Plot Embarked counts
plt.subplot2grid(fig_dims, (1, 1))
df['Embarked'].value_counts().plot(kind='bar', title='Ports of Embarkation Counts')

# Plot the Age histogram
plt.subplot2grid(fig_dims, (2, 0))
df['Age'].hist()
plt.title('Age Histogram')




# Pclass
pclass_xt = pd.crosstab(df['Pclass'], df['Survived'])
pclass_xt




# Normalize the cross tab to sum to 1:
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)
pclass_xt_pct




pclass_xt_pct.plot(kind='bar', stacked=True, title='Survival Rate by Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')




# Sex
df['Sex'].unique()




sexes = sorted(df['Sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
genders_mapping




df['Sex_Val'] = df['Sex'].map(genders_mapping).astype(int)
df.head()




sex_val_xt = pd.crosstab(df['Sex_Val'], df['Survived'])
sex_val_xt_pct = sex_val_xt.div(sex_val_xt.sum(1).astype(float), axis=0)
sex_val_xt_pct.plot(kind='bar', stacked=True, title='Survival Rate by Gender')




# Embarked
df[df['Embarked'].isnull()]




# Get the unique values of Embarked
embarked_locs = sorted(df['Embarked'].unique())
embarked_locs_mapping = dict(zip(embarked_locs, range(0, len(embarked_locs) + 1)))
embarked_locs_mapping




df['Embarked_Val'] = df['Embarked'].map(embarked_locs_mapping).astype(int)
df.head()




df['Embarked_Val'].hist(bins=len(embarked_locs), range=(0, 3))
plt.title('Port of Embarkation Histogram')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.show()




# Since the vast majority of passengers embarked in 'S': 3, 
# we assign the missing values in Embarked to 'S':
if len(df[df['Embarked'].isnull()] > 0):
    df.replace({'Embarked_Val' : { embarked_locs_mapping[np.nan] : embarked_locs_mapping['S']}}, 
               inplace=True)




embarked_locs = sorted(df['Embarked_Val'].unique())
embarked_locs




embarked_val_xt = pd.crosstab(df['Embarked_Val'], df['Survived'])
embarked_val_xt_pct =     embarked_val_xt.div(embarked_val_xt.sum(1).astype(float), axis=0)
embarked_val_xt_pct.plot(kind='bar', stacked=True)
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival Rate')




# Age
df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']].head()




# To keep Age in tact, make a copy of it called AgeFill 
# that we will use to fill in the missing ages:
df['AgeFill'] = df['Age']

# Populate AgeFill
df['AgeFill'] = df['AgeFill'].groupby([df['Sex_Val'], df['Pclass']])                         .apply(lambda x: x.fillna(x.median()))




len(df[df['AgeFill'].isnull()])




df[df['AgeFill'].isnull()][['Sex', 'Pclass', 'Age']].head()




# Set up a grid of plots
fig, axes = plt.subplots(2, 1, figsize=(10,10))

# Histogram of AgeFill segmented by Survived
df1 = df[df['Survived'] == 0]['Age']
df2 = df[df['Survived'] == 1]['Age']
max_age = max(df['AgeFill'])
axes[0].hist([df1, df2], 
             bins=max_age / 10, # bin_size
             range=(1, max_age), 
             stacked=True)
axes[0].legend(('Died', 'Survived'), loc='best')
axes[0].set_title('Survivors by Age Groups Histogram')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Scatter plot Survived and AgeFill
axes[1].scatter(df['Survived'], df['AgeFill'])
axes[1].set_title('Survivors by Age Plot')
axes[1].set_xlabel('Survived')
axes[1].set_ylabel('Age')




passenger_classes = sorted(df['Pclass'].unique())
for pclass in passenger_classes:
    df.AgeFill[df.Pclass == pclass].plot(kind='kde')
plt.title('Age Density Plot by Passenger Class')
plt.xlabel('Age')
plt.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')




df['Fare'].hist(bins=25)






