#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_validate, cross_val_score, GroupKFold, KFold
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RandomizedSearchCV




train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")




train.head()




plt.figure(figsize=(10,10))
sns.heatmap(train.corr(), annot=True, cmap="Blues")




plt.figure(figsize=(9,6))
sns.barplot(x='Sex',y='Survived',data=train)




plt.figure(figsize=(9,6))
sns.barplot(x='Pclass', y='Survived', data=train)




train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)




new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

new_data_train.head()




new_data_train.isnull().sum().sort_values(ascending=False).head(10)




new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)

new_data_test.isnull().sum().sort_values(ascending=False).head(10)




new_data_train['Fare'].fillna(new_data_train['Fare'].mean(), inplace=True)
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)

new_data_test.isnull().sum().sort_values(ascending=False).head(10)




x = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']




model = DecisionTreeClassifier(max_depth=3)
model.fit(x, y)

accuracy = round(model.score(x, y) * 100, 2)

print(accuracy)




SEED = 301
np.random.seed(SEED)

parameters = {
    "max_depth": [3, 5],
    "min_samples_split": [32, 64, 128],
    "min_samples_leaf": [32, 64, 128],
    "criterion": ["gini", "entropy"]
}

searchGrid = RandomizedSearchCV(DecisionTreeClassifier(), parameters, n_iter=16, cv=KFold(n_splits=5, shuffle=True), random_state=SEED)
searchGrid.fit(x, y)

scores = cross_val_score(searchGrid, x, y, cv=KFold(n_splits=5, shuffle=True))

average = scores.mean()
deviation  = scores.std()

best_estimator = searchGrid.best_estimator_
print(best_estimator)
print("Accuracy average %.2f" % round(average * 100, 2))
print("Deviation [%.2f, %.2f]" % (round((average - 2 * deviation) * 100, 2), round((average + 2 * deviation) * 100, 2)))




model = DummyClassifier()
results = cross_validate(model, x, y, cv=10, return_train_score=False)
accuracy = round(results['test_score'].mean() * 100, 2)

print(accuracy)




model_random_forest = RandomForestClassifier(n_jobs=10, random_state=0)
model_random_forest.fit(x, y)

accuracy = round(model_random_forest.score(x, y) * 100, 2)

print(accuracy)




submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = model_random_forest.predict(new_data_test)

submission.to_csv('submission.csv', index=False)

