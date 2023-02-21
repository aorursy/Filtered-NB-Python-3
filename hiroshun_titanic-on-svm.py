#!/usr/bin/env python
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




best_C = 0.3
best_gamma = 0.1




df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')




n_all_train = len(df_train) #891
n_test = len(df_test) #418




df_all = pd.concat([df_train, df_test], ignore_index=True)




all_drop = df_all.drop(["Cabin", "Fare", "Ticket"], axis=1)




alone = all_drop.apply(lambda p: 1 if p.SibSp + p.Parch == 0 else 0, axis=1)
all_drop['alone'] = alone

all_drop2 = all_drop.drop(["SibSp", "Parch"], axis=1)




def get_familyname(name):
    l = name.split(' ')
    return l[0]
    
familyname = all_drop2.Name.apply(get_familyname)




name_title = ['Mr.', 'Master.', 'Mrs.', 'Miss.']
for nt in name_title:
    all_drop2[nt] = all_drop2.apply(lambda passenger: int(nt in passenger.Name), axis=1)




all_drop2['familyname'] = familyname
all_drop2['num_family_members'] = all_drop.SibSp + all_drop.Parch

all_drop3 = all_drop2.drop('Name', axis=1)




all_drop3 = pd.concat((all_drop3, pd.get_dummies(all_drop2['Sex'])), axis=1)
all_drop3 = all_drop3.drop(['Sex', 'female'], axis=1)

all_drop3 = pd.concat((all_drop3, pd.get_dummies(all_drop3['Embarked'])), axis=1)
all_drop3 = all_drop3.drop(['Embarked', 'S'], axis=1)

all_drop3 = pd.concat((all_drop3, pd.get_dummies(all_drop3['Pclass'])), axis=1)
all_drop3 = all_drop3.drop(['Pclass', 1], axis=1)
all_drop3.rename(columns={2: "class2", 3: "class3"}, inplace=True)




all_drop3.head()




age_learndata = all_drop3.loc[all_drop3.Age.notnull()].drop(['PassengerId', 'Survived', 'familyname'], axis=1)
age_learndata_X = age_learndata.drop(['Age'], axis=1).as_matrix()
age_learndata_y = age_learndata.Age

age_predictdata_X = all_drop3.loc[all_drop3.Age.isnull()].drop(['PassengerId', 'Survived', 'familyname', 'Age'], axis=1)




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

age_learndata_X_std = sc.fit_transform(age_learndata_X)
age_predictdata_X_std = sc.transform(age_predictdata_X)




from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(age_learndata_X_std, age_learndata_y)




age_predicteddata = knr.predict(age_predictdata_X_std)

all_drop3.loc[all_drop3.Age.isnull(), 'Age'] = knr.predict(age_predictdata_X_std)




train = all_drop3[:n_all_train]
test = all_drop3[n_all_train:]




family_group = train.groupby(['num_family_members', 'familyname'])




family_survived_rate =     pd.concat([family_group.mean().Survived.rename('family_rate'), family_group.count().Survived.rename('count')], axis=1)




family_survived_rate.reset_index(inplace=True)




train_drop = pd.merge(train, family_survived_rate, on=['num_family_members', 'familyname'], suffixes=('', '_family'))




train_drop.info()




survived_rate = train_drop['Survived'].mean()




def recalc_rate(p):
    if p['count'] == 1:
        return survived_rate
    else:
        return (p['family_rate'] * p['count'] - p['Survived']) / (p['count'] - 1)
        
train_drop['family_rate'] = train_drop.apply(recalc_rate, axis=1).sort_values()




train_drop2 = train_drop.drop(['familyname', 'count'], axis=1)
family_survived_rate.drop('count', inplace=True, axis=1)




np.abs(train_drop2.corr()).style.background_gradient()




train_drop2 = train_drop2.reindex(np.random.permutation(train_drop2.index)).reset_index(drop=True)




X_train = train_drop2.drop(['Survived', 'PassengerId'], axis=1).as_matrix()
y_train = train_drop2.Survived.tolist()




X_train_std = sc.fit_transform(X_train)




from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=best_C, gamma=best_gamma, random_state=0)
svm.fit(X_train_std, y_train)




test2 = pd.merge(test, family_survived_rate, on=['num_family_members', 'familyname'], how='left')
test2.family_rate.fillna(survived_rate, inplace=True)
test3 = test2.drop('familyname', axis=1)




test3.info()




X_test = test3.drop(['Survived', 'PassengerId'], axis=1).as_matrix()




X_test_std = sc.transform(X_test)




y_test = svm.predict(X_test_std)




test3.loc[:,"Survived"] = y_test




output_csv = test3.loc[:, ["PassengerId", "Survived"]]
output_csv.Survived = output_csv.Survived.astype(int)




output_csv.to_csv("submisson.csv", index=False)

