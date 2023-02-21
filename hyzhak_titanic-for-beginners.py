#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns
import sklearn
from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree
import subprocess

get_ipython().run_line_magic('matplotlib', 'inline')




train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]




train_df.head()




train_df.tail()




print('# features:')
print(train_df.columns.values)
print('_'*40)
print('# data types:')
train_df.info()
print('_'*40)
test_df.info()




# numberical features
train_df.describe()




# categorical features
train_df.describe(include=['O'])




def chance_to_survive_by_feature(feature_name):
    return train_df[[feature_name, 'Survived']]        .groupby([feature_name])        .mean()        .sort_values(by='Survived', ascending=False)    

chance_to_survive_by_feature('Pclass')




chance_to_survive_by_feature('Sex')




chance_to_survive_by_feature('SibSp')




chance_to_survive_by_feature('Parch')




g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20);




grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();




ordered_embarked = train_df.Embarked.value_counts().index

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend();




grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend();




print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print("After ", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)




for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])




for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',                                                  'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()




title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()




train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape




for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()




grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();




guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)




for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()




train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()




for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[[
    'FamilySize', 
    'Survived',
]].groupby([
    'FamilySize'
], as_index=False)\
.mean()\
.sort_values(by='Survived', ascending=False)




for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[[
    'IsAlone', 
    'Survived',
]]\
.groupby(['IsAlone'], as_index=False)\
.mean()




train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()




for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)




freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[[
    'Embarked', 
    'Survived',
]]\
.groupby(['Embarked'], as_index=False)\
.mean()\
.sort_values(by='Survived', ascending=False)




for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()




test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()




train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[[
    'FareBand', 
    'Survived',
]]\
.groupby(['FareBand'], as_index=False)\
.mean()\
.sort_values(by='FareBand', ascending=True)




for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)




X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape




models = []

models.append({
    'classifier': linear_model.LogisticRegression,
    'name': 'Logistic Regression',
})
models.append({
    'classifier': svm.SVC,
    'name': 'Support Vector Machines',
})
models.append({
    'classifier': neighbors.KNeighborsClassifier,
    'name': 'k-Nearest Neighbors',
    'args': {
        'n_neighbors': 3,
    },
})
models.append({
    'classifier': naive_bayes.GaussianNB,
    'name': 'Gaussian Naive Bayes',
})
models.append({
    'classifier': linear_model.Perceptron,
    'name': 'Perceptron',
    'args': {
        'max_iter': 5,
        'tol': None,
    },
})
models.append({
    'classifier': svm.LinearSVC,
    'name': 'Linear SVC',
})
models.append({
    'classifier': linear_model.SGDClassifier,
    'name': 'Stochastic Gradient Descent',
    'args': {
        'max_iter': 5,
        'tol': None,
    },
})
models.append({
    'classifier': tree.DecisionTreeClassifier,
    'name': 'Decision Tree',
})
models.append({
    'classifier': ensemble.RandomForestClassifier,
    'name': 'Random Forest',
    'args': {
        'n_estimators': 100,
    },
})

#acc_log




def process_model(model_desc):
    Model = model_desc['classifier']
    model = Model(**model_desc.get('args', {}))
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = round(model.score(X_train, Y_train) * 100, 2)
    return {
        'name': model_desc['name'],
        'accuracy': accuracy,
        'model': model,
    }

models_result = list(map(process_model, models))
models_result = sorted(models_result, key=lambda res: res['accuracy'], reverse=True)

#print(models_result)

# plot bars
models_result_df = pd.DataFrame(models_result, columns=['accuracy', 'name'])
ax = sns.barplot(data=models_result_df, x='accuracy', y='name')
ax.set(xlim=(0, 100))

# show table
models_result_df




# use keras (tensorflow) for full-convolutional deep NN




# submission.to_csv('../output/submission.csv', index=False)
the_best_result = models_result[0]
Y_pred = the_best_result['model'].predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': Y_pred,
})
submission.to_csv('submission.csv', index=False)

