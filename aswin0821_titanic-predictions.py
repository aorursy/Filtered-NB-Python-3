#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()




test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()




print(train.shape, test.shape)




train.isnull().sum()




train['Survived'].value_counts()




train.describe(include = 'O')




train.describe()




test.describe()




train[(train['Survived'] == 1) & (train['Sex'] == 'female')].shape[0]/train[train['Sex'] == 'female'].shape[0]
#74% of the female travellers survived




train[(train['Survived'] == 1) & (train['Sex'] == 'male')].shape[0]/train[train['Sex'] == 'male'].shape[0]
#18% of the male travellers survived




g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=30)




grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

#Number of people who had ticket class 1 (Pclass = 1) survived more than the number of those having Pclass = 1 died
#Number of people who had ticket class 2 (Pclass = 2) were more or less equiprobable of surviving or dying
#Number of people who had ticket class 3 (Pclass = 3) had a higher chance of dying




grid = sns.FacetGrid(train, row='Embarked', height = 2.2, aspect = 1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

#Males had a higher chance of surviving had they embarked in Port C
#Males were almost sure of dying had they embarked in Port Queenstown
#Females had a higher chance of surviving had they embarked in Port Queenstown followed by Southampton




grid = sns.FacetGrid(train, row='Embarked', col='Survived', height = 2.2, aspect = 1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

#On an average, passengers paying higher fare had a higher chance of survival




print("Before: ", train.shape, test.shape)
train = train.drop(['Ticket', 'Cabin'], axis = 1)
test = test.drop(['Ticket', 'Cabin'], axis = 1)
print("After: ", train.shape, test.shape)




train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)
combine = [train, test]
train.shape, test.shape




for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head()




grid = sns.FacetGrid(train, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()




guess_ages = np.zeros((2,3))
guess_ages
#Guess values for sex, Pclass




for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train.head()




#Creating age bands for different age groups and determining their correlation with survival.
train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)




for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train.head()




train = train.drop(['AgeBand'], axis=1)
combine = [train, test]
train.head()




#Combining the features Parch(#Parents/Children) and SibSp(#Siblings/Spouses) and adding 1 to it(The individual himself)
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)




#Defining whether a person travelled on the ship without any family members
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()




#Dropping the features SibSp, Parch, FamilySize
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train, test]

train.head()




train['Embarked'].value_counts()




freq = train['Embarked'].mode()
freq[0]




train['Embarked'] = train['Embarked'].fillna(freq[0])




#Converting categorical features to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()




#Fare column in test dataset had one missing value. We will replace it with the mean
test['Fare'].fillna(test['Fare'].dropna().mean(), inplace=True)




#creating a fare band so as to convert them into single numeric values
train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)




#Converting fare bands into numerical values
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis = 1)
combine = [train, test]
    
train.head()




X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, y_train.shape, X_test.shape




from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
Y_pred = lr.predict(X_test)




acc = round(lr.score(X_train, y_train) * 100, 2)
acc




coeff = pd.DataFrame(train.columns.delete(0))
coeff.columns = ['Feature']
coeff["Correlation"] = pd.Series(lr.coef_[0])

coeff.sort_values(by='Correlation', ascending=False)

#We can see that Sex has the highest correlation to survival. AS the value of Sex increases from 0 to 1(male to female), the chances of survival
#increases. Inversely, as the PClass values change from 1 to 3, survival rate decreases




#Using SVM
from sklearn.svm import SVC
clf = SVC(gamma = 'auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)




accSvm = round(clf.score(X_train, y_train) * 100, 2)
accSvm




from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)




accKNN = round(neigh.score(X_train, y_train) * 100, 2)
accKNN




from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()
clf2.fit(X_train, y_train)
y_pred = clf.predict(X_test)




accNB = round(clf2.score(X_train, y_train) * 100, 2)
accNB




from sklearn.tree import DecisionTreeClassifier
clft = DecisionTreeClassifier(random_state = 0)
clft.fit(X_train, y_train)
y_pred = clft.predict(X_test)




accDT = round(clft.score(X_train, y_train) * 100, 2)
accDT




from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth = 3, random_state = 0)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)




accRF = round(forest.score(X_train, y_train) * 100, 2)
accRF




models = pd.DataFrame({
   'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Decision Tree'],
   'Score': [accSvm, accKNN, acc, accRF, accNB, accDT]})
models.sort_values(by='Score', ascending=False)

#We see that Decision Tree performs best on the training set




submission = pd.DataFrame({ "PassengerId": test["PassengerId"], "Survived": y_pred})
submission.to_csv('submission.csv', index=False)




get_ipython().system('ls')






