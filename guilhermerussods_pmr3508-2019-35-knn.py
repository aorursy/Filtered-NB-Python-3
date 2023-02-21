#!/usr/bin/env python
# coding: utf-8



import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing

from sklearn.metrics import accuracy_score




sns.set_style('darkgrid')
pd.set_option('Display.max_columns', None)




train = pd.read_csv('1 - Dados/train_data.csv')
test = pd.read_csv('1 - Dados/test_data.csv')




train.shape, test.shape




train.head(3)




# Identify NaN 
train.replace('?', np.nan, inplace=True)
# Remove Id column
train.set_index('Id', inplace=True)




# Identify missing data and datatypes 
train.info()




missing_data_cols = ['workclass', 'occupation', 'native.country']
for col in missing_data_cols:
    nunique = train[col].nunique()
    mostfreq = train[col].value_counts().index[0]
    freq = train[col].value_counts().iloc[0]
    print('Coluna : {}\n    Nº de únicos  : {}\n    Nome          : {}\n    Mais frequente: {}\n'.format(col, nunique, mostfreq, freq))




# The only columns that doesn't present any kind of predominance is occupation

# Identify occupation by the type of worckclass
train[~train.occupation.notna()].workclass.value_counts()




# Almost all missing occupation data receive under 50K
train[~train.occupation.notna()].income.value_counts()




train[train.workclass=='Private'].occupation.value_counts()




# Para workclass e native.country podemos preencher pela moda devido à frequência
train['workclass'].fillna(train['workclass'].value_counts().index[0], inplace=True)
train['native.country'].fillna(train['native.country'].value_counts().index[0], inplace=True)
# As we would need almost to create a model to identify the missing occupation data
# we will fill by "unkown"
train.occupation.fillna('other', inplace=True)




train.head(3)




def quartile_class(series):
    q3, q2, q1 = series.quantile(q=[0.25, 0.5, 0.75])
    outp = []
    for val in series:
        if val>q1:
            outp.append('Q1')
        elif val>q2:
            outp.append('Q2')
        elif val>q3:
            outp.append('Q3')
        else:
            outp.append('Q4')
    return outp




fig, ax = plt.subplots(figsize=(20, 6))

categ_cols = ['workclass', 'education', 'marital.status', 'occupation']
for i, col in enumerate(categ_cols):
    axi = plt.subplot(2,2, i+1)
    h = train[train.income=='>50K'][col].value_counts().reset_index()
    l = train[train.income=='<=50K'][col].value_counts().reset_index()
    aux = l.merge(h, on='index', how='left')
    aux.plot(kind='bar', x='index', y=col+'_x', ax=axi)
    aux.plot(kind='bar', x='index', y=col+'_y', color='firebrick', alpha=0.6, ax=axi)
    plt.xticks(rotation=80);
    plt.title(col)
    plt.legend(['<=50K', ' >50K'])
    plt.xlabel('')
plt.subplots_adjust(hspace = 0.7)




# It is clear that some categorys concentrate high income observations
# It happens clearly on education, occupation and marital.status




fig, ax = plt.subplots(figsize=(20, 6))

categ_cols = ['relationship', 'race', 'sex', 'native.country']
for i, col in enumerate(categ_cols):
    axi = plt.subplot(2,2, i+1)
    h = train[train.income=='>50K'][col].value_counts().reset_index()
    l = train[train.income=='<=50K'][col].value_counts().reset_index()
    aux = l.merge(h, on='index', how='left')
    aux.plot(kind='bar', x='index', y=col+'_x', ax=axi)
    aux.plot(kind='bar', x='index', y=col+'_y', color='firebrick', alpha=0.6, ax=axi)
    plt.xticks(rotation=80);
    plt.title(col)
    plt.legend(['<=50K', '>50K'])
    plt.xlabel('')
plt.subplots_adjust(hspace = 0.7)




# The same happens for the 4 charts above for all collumns




# Let's try to understand which of those categorys concentrate high income observations
# and which concentrate low income

categ_cols = ['workclass', 'marital.status', 'occupation', 'sex',
               'relationship', 'race', 'native.country', 'education']
incomeAnalysis = {}
for col in categ_cols:
    df1 = train[train.income=='<=50K'].groupby([col], as_index=False).agg({'income':'count'})
    df2 = train[train.income=='>50K'].groupby([col], as_index=False).agg({'income':'count'})
    df = df1.merge(df2, on=col, how='left', suffixes=('<=50K', '>50K'))
    df['Percent_of_HighIncome'] = df['income>50K']/df['income<=50K']
    df[col+'_quartile'] = quartile_class(df['Percent_of_HighIncome'])
    df.sort_values(by='Percent_of_HighIncome', ascending=False, inplace=True)
    incomeAnalysis[col] = df.sort_values(by='Percent_of_HighIncome', ascending=False)




train[(train.workclass=='Self-emp-inc')&(train.income=='>50K')].shape[0],train[(train.workclass=='Self-emp-inc')&(train.income=='<=50K')].shape[0]




#Example
incomeAnalysis['workclass']




# Reduce number of categories in categorical columns by using the quartiles
train2 = train.copy()
for col in categ_cols:
    train2 = train2.merge(incomeAnalysis[col][[col, col+'_quartile']], on=col, how='left')




train2.head()




train.head(2)




fig, ax = plt.subplots(figsize=(20, 8))

numerical_cols = ['age', 'hours.per.week']
for i, col in enumerate(numerical_cols):
    axi = plt.subplot(2,1, i+1)
    h = train[train.income=='>50K'][col].value_counts().reset_index().sort_values(by='index')
    l = train[train.income=='<=50K'][col].value_counts().reset_index().sort_values(by='index')
    aux = l.merge(h, on='index', how='left')
    aux.plot(kind='bar', x='index', y=col+'_x', ax=axi)
    aux.plot(kind='bar', x='index', y=col+'_y', color='firebrick', alpha=0.6, ax=axi)
    plt.xticks(rotation=80);
    plt.title(col)
    plt.legend(['<=50K', '>50K'])
    plt.xlabel('')
    
plt.subplots_adjust(hspace = 0.4)




train3 = train2.copy()
age = pd.cut(train['age'], bins = [-1, 25, 30, 35, 52, 60, 75, 200], labels = [0, 1, 2, 3, 4, 5, 6])
train3['age_clusters'] = list(age)

hours_week = pd.cut(train['hours.per.week'], bins = [-1, 25, 40, 60, 200], labels = [0, 1, 2, 3])
train3['hours_week_clusters'] = list(hours_week)




fig, ax = plt.subplots(figsize=(22, 8))

numerical_cols = ['capital.gain', 'capital.loss', 'fnlwgt']
for i, col in enumerate(numerical_cols):
    axi = plt.subplot(1,3, i+1)    
    plt.hist(train[train.income=='<=50K'][col])
    plt.hist(train[train.income=='>50K'][col], color='firebrick', alpha=0.5)
    plt.xticks(rotation=80);
    plt.title(col)
    plt.legend(['<=50K', '>50K'])
    plt.xlabel('')

plt.subplots_adjust(hspace = 0.7)




median = np.median(train[train['capital.gain'] > 0]['capital.gain'])
capital_gain_clusters = pd.cut(train['capital.gain'],
             bins = [-1, 0, median, train['capital.gain'].max()+1],
             labels = [0, 1, 2])
train3['capital_gain_clusters'] = list(capital_gain_clusters)

median = np.median(train[train['capital.loss'] > 0]['capital.loss'])
capital_loss_clusters = pd.cut(train['capital.loss'],
             bins = [-1, 0, median, train['capital.loss'].max()+1],
             labels = [0, 1, 2])
train3['capital_loss_clusters'] = list(capital_loss_clusters)




train3.head()




train4 = train2[['age', 'workclass_quartile', 'fnlwgt', 'education_quartile',
                 'marital.status_quartile', 'occupation_quartile', 'relationship_quartile',
                 'race_quartile', 'sex_quartile', 'capital.gain', 
                 'capital.loss', 'hours.per.week', 'native.country_quartile', 
                 'income']]
train4 = train4.apply(preprocessing.LabelEncoder().fit_transform)




get_ipython().run_cell_magic('time', '', "x = train4.drop(columns='income')\ny = train4['income']\n\nx_train, x_test, y_train, y_test = train_test_split(x, y,\n                                                    test_size=0.1,\n                                                    random_state=42)\nknn = KNeighborsClassifier(n_neighbors=20)\nknn.fit(x_train, y_train)\n\ny_pred = knn.predict(x_test)\nprint(accuracy_score(y_test, y_pred))")




get_ipython().run_cell_magic('time', '', "x = train4.drop(columns='income')\ny = train4['income']\n\nx_train, x_test, y_train, y_test = train_test_split(x, y,\n                                                    test_size=0.1,\n                                                    random_state=42)\n\nknn_grid_search_dict = {'n_neighbors'   :[20, 30, 40],\n                        'weights'       :['uniform', 'distance'],\n                        'algorithm'     :['auto'],\n                        'leaf_size'     :[20, 30, 40],\n                        'p'             :[1, 2],\n                        'n_jobs'        :[-1]}\n\n\nknn = GridSearchCV(estimator = knn, \n                   cv = 5,\n                   param_grid = knn_grid_search_dict, \n                   scoring = 'accuracy')\nprint('ok')\nknn.fit(x_train, y_train)\nprint('ok2')\n\nprint(knn.best_params_)\ny_pred = knn.predict(x_test)\n\nprint(accuracy_score(y_test, y_pred))\n#pd.to_pickle(knn, 'knnclf_tunned.pickle')")




test.replace('?', np.nan, inplace=True)
test.set_index('Id', inplace=True)

test['workclass'].fillna(test['workclass'].value_counts().index[0], inplace=True)
test['native.country'].fillna(test['native.country'].value_counts().index[0], inplace=True)
test.occupation.fillna('other', inplace=True)

test2 = test.copy()
for col in categ_cols:
    test2 = test2.merge(incomeAnalysis[col][[col, col+'_quartile']], on=col, how='left')




test3 = test2[['age', 'workclass_quartile', 'fnlwgt', 'education_quartile',
               'marital.status_quartile', 'occupation_quartile', 'relationship_quartile',
               'race_quartile', 'sex_quartile', 'capital.gain', 
               'capital.loss', 'hours.per.week', 'native.country_quartile']]
test3 = test3.apply(preprocessing.LabelEncoder().fit_transform)




#knn = pd.read_pickle('knnclf_tunned.pickle')
y_pred = knn.predict(test3)
test['income'] = y_pred
test['income'] = test['income'].replace({0:'<=50K', 1:'>50k'})




test[['income']].to_csv('sample_submission.csv')

