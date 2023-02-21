#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBClassifier




train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
print('\n ---------------------------------- \n')
test.info()




train_test = pd.concat([train, test])




sns.countplot(data=train_test, x='Pclass')
plt.show()




sns.countplot(data=train, x='Pclass', hue='Sex')
plt.show()




f, ax = plt.subplots(nrows=2, sharex=True)
sns.countplot(data=train, x='Pclass', hue='Embarked', ax=ax[0], hue_order=['S','C','Q'])
sns.countplot(data=test, x='Pclass', hue='Embarked', ax=ax[1], hue_order=['S','C','Q'])
ax[0].set_xlabel('train')
ax[1].set_xlabel('test')
plt.show()




print(train.groupby(['Pclass']).mean()['Survived'])
print('\n ------------------- \n')
print(train.groupby(['Pclass','Sex']).mean()['Survived'].unstack())
print('\n ------------------- \n')
print(train.groupby(['Pclass','Embarked']).mean()['Survived'].unstack())




# Fill the missing values
train['Embarked'] = train.Embarked.fillna(train.Embarked.mode())




train_test = pd.concat([train,test])
sns.countplot(data=train_test, x='Embarked', order=['S','C','Q'])
plt.show()




sns.countplot(data=train_test, x='Embarked', hue='Sex')
plt.show()




sns.countplot(data=train_test, x='Embarked', hue='Pclass')
plt.show()




print(train.groupby(['Embarked']).mean()['Survived'])
print('\n ------------------- \n')
print(train.groupby(['Embarked','Sex']).mean()['Survived'].unstack())




test['Fare'] = test.Fare.fillna(train_test.Fare.median())




train_test = pd.concat([train,test])
f, ax = plt.subplots(1)
sns.distplot(train_test.Fare, kde=None)
ax.set_xscale('log')
plt.show()




print(train_test.Fare.describe())




sns.pointplot(data=train_test, x='Pclass', y='Fare', hue='Sex')
plt.suptitle('Fare - Pclass')
plt.show()




sns.pointplot(data=train_test, x='Embarked', y='Fare', order=['S','C','Q'], hue='Sex')
plt.suptitle('Fare - Embarked')
plt.show()




print(train.groupby('Survived').mean()['Fare'])
print('\n ------------------- \n')
print(train.groupby(['Survived','Sex']).mean()['Fare'].unstack())




train['FamilySize'] = train.SibSp + train.Parch + 1 
test['FamilySize'] = test.SibSp + test.Parch + 1  
train_test = pd.concat([train,test])
f, ax = plt.subplots(nrows=2)
sns.countplot(data=train, x='FamilySize', hue='Survived', ax=ax[0])
sns.countplot(data=train, x='FamilySize', hue='Survived', order=[5,6,7,8,11], ax=ax[1])
ax[0].set_xlabel('')
plt.show()




train_test[train_test.FamilySize == 8].drop(['PassengerId','Embarked','Cabin'], axis=1)




train_test[train_test.FamilySize == 11].drop(['PassengerId','Embarked','Cabin'], axis=1)




train_test[train_test.FamilySize == 7].sort_values(by='Name').drop(['PassengerId','Embarked','Cabin'], axis=1)




train_test[train_test.FamilySize == 6].sort_values(by='Name').drop(['PassengerId','Embarked','Cabin'], axis=1)




train.loc[train.Ticket == 'LINE', 'Ticket'] = 'LINE '
train['tk_head'] = train.Ticket.apply(lambda x: re.search(r'[A-Za-z]+.*\s',x).group() if re.search(r'[A-Z]+.*\s',x) else '')
test['tk_head'] = test.Ticket.apply(lambda x: re.search(r'[A-Za-z]+.*\s',x).group() if re.search(r'[A-Z]+.*\s',x) else '')




train['tk_head'] = train.tk_head.apply(lambda x: re.sub(r'\.?/?\s?','',x).upper())
test['tk_head'] = test.tk_head.apply(lambda x: re.sub(r'\.?/?\s?','',x).upper())




train_test = pd.concat([train,test])
train_test.groupby(['tk_head','Pclass']).mean()[['Fare']].join(train_test.groupby(['tk_head','Pclass']).count()['PassengerId'])




train['tk_num'] = train.Ticket.apply(lambda x: re.split(r'\s',x)[-1])
test['tk_num'] = test.Ticket.apply(lambda x: re.split(r'\s',x)[-1])




train_test = pd.concat([train,test])
tk_count = train_test.groupby(['tk_head','tk_num']).count()[['PassengerId']]
tk_count[tk_count.PassengerId>1]




train_test.loc[train_test.Ticket == 'PC 17608'].drop(['PassengerId','Ticket'], axis=1)




train_test.loc[train_test.Ticket == 'W./C. 6608'].drop(['PassengerId','Ticket'], axis=1)




train_test.loc[(train_test.Ticket == 'STON/O 2. 3101273') | (train_test.Ticket == 'STON/O 2. 3101274') |(train_test.Ticket == 'STON/O 2. 3101275')].drop(['PassengerId','Ticket'], axis=1)




train_test.loc[(train_test.Ticket == '112050') 
               | (train_test.Ticket == '112051') 
               |(train_test.Ticket == '112052')
               |(train_test.Ticket == '112053')].sort_values(by='tk_num').drop(['PassengerId','Ticket'], axis=1)




i = 0 # use i as the unique label of Group
for (head,num) in list(tk_count[tk_count.PassengerId > 1].index):
    train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == num), 'Group'] = i
    i += 1




index_dict = {head:[] for (head,num) in list(tk_count[tk_count.PassengerId == 1].index)}
for (head,num) in list(tk_count[tk_count.PassengerId == 1].index):
    index_dict[head].append(int(num))

i-=1
del head, num

for index in index_dict.items():
    head = index[0]
    num_list = sorted(index[1])
    flag = False
    for (j,j_value) in enumerate(num_list):
        if j < len(num_list)-1 :
            j_next_value = num_list[j+1]
            if  j_next_value - j_value == 1:
                j_Embarked = train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == str(j_value)),'Embarked'].values
                j_next_Embarked = train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == str(j_next_value)),'Embarked'].values
                if j_Embarked == j_next_Embarked:
                    j_Pclass = train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == str(j_value)),'Pclass'].values
                    j_next_Pclass = train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == str(j_next_value)),'Pclass'].values
                    if j_Pclass == j_next_Pclass:
                        j_Fare = train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == str(j_value)),'Fare'].values
                        j_next_Fare = train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == str(j_next_value)),'Fare'].values
                        if j_Fare == j_next_Fare:
                            if not flag: 
                                i += 1
                            train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == str(j_value)), 'Group'] = i
                            train_test.loc[(train_test.tk_head == head) & (train_test.tk_num == str(j_next_value)), 'Group'] = i
                            flag = True
                        else:
                            flag = False
                    else:
                        flag = False
                else:
                    flag = False
            else:
                flag = False




train  = pd.merge(train, train_test[['PassengerId','Group']], on='PassengerId')
test  = pd.merge(test, train_test[['PassengerId','Group']], on='PassengerId')




train.Cabin.fillna('0', inplace=True)
test.Cabin.fillna('0', inplace=True)
train['Cabin_head'] = train.Cabin.apply(lambda x: re.search(r'[A-Za-z]*',x).group())
test['Cabin_head'] = test.Cabin.apply(lambda x: re.search(r'[A-Za-z]*',x).group())




train_test = pd.concat([train,test])
train_test.groupby(['tk_head','Cabin_head']).mean()[['Fare']].join(train_test.groupby(['tk_head','Cabin_head']).count()['PassengerId']).unstack()




pd.merge(train_test.groupby(['Cabin_head']).mean()[['Fare']].reset_index() , 
         train_test.groupby(['Cabin_head']).median()[['Fare']].reset_index(), 
         on='Cabin_head', suffixes=['_average','_median']).\
merge(train_test.groupby(['Cabin_head']).count()['PassengerId'].reset_index())




train['Cabin'] = train.Cabin_head.map({'A':3, 'B':1, 'C':1, 'D':2, 'E':2, 'F':4, 'G':4, 'T':3, '':0})
test['Cabin'] = test.Cabin_head.map({'A':3, 'B':1, 'C':1, 'D':2, 'E':2, 'F':4, 'G':4, 'T':3, '':0})
train.drop('Cabin_head',axis=1,inplace=True)
test.drop('Cabin_head',axis=1,inplace=True)




# EMBARKED
train = train.join(pd.get_dummies(train.Embarked, prefix='Embarked_',drop_first=True))
train.drop('Embarked',axis=1,inplace=True)
test = test.join(pd.get_dummies(test.Embarked, prefix='Embarked_',drop_first=True))
test.drop('Embarked',axis=1,inplace=True)




# TICKET
train.drop(['Ticket','tk_head','tk_num'],axis=1,inplace=True)
test.drop(['Ticket','tk_head','tk_num'],axis=1,inplace=True)




train.loc[train.Group.isnull(),'IsAlone'] = 1
train.loc[train.Group.notnull(),'IsAlone'] = 0
test.loc[train.Group.isnull(),'IsAlone'] = 1
test.loc[train.Group.notnull(),'IsAlone'] = 0




train.loc[train.Group.isnull(),'Group'] = train_test.Group.max() + 1
test.loc[test.Group.isnull(),'Group'] = train_test.Group.max() + 1




title_map = {'Don':'Mr','Dona':'Mrs','Mme':'Mrs','Mlle':'Miss','Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Ms':'Miss','Master':'Master',             'Dr':'High','Lady':'High','Sir':'High','Countess':'High', 'Jonkheer':'High',            'Rev':'M&R','Major':'M&R', 'Col':'M&R', 'Capt':'M&R'}
train['Title'] = train.Name.apply(lambda x: re.search(r'\s[A-Za-z]+\.',x).group().split('.')[0].strip()                                  if re.search(r'\s[A-Za-z]+\.',x).group() else '').map(title_map)
test['Title'] = test.Name.apply(lambda x: re.search(r'\s[A-Za-z]+\.',x).group().split('.')[0].strip()                                if re.search(r'\s[A-Za-z]+\.',x).group() else '').map(title_map)




train.groupby('Title').mean()[['Survived']].reset_index().merge(train.groupby('Title').count()[['PassengerId']].reset_index(), on='Title').sort_values(by='Survived')




train = train.join(pd.get_dummies(train.Title, prefix='Title_', drop_first=True))
train.drop(['Name','Title'],axis=1,inplace=True)
test = test.join(pd.get_dummies(test.Title, prefix='Title_', drop_first=True))
test.drop(['Name','Title'],axis=1,inplace=True)




train_test = pd.concat([train,test])
sns.countplot(data=train, x='Sex', hue='Survived')
plt.show()




g = sns.FacetGrid(train_test,col='Sex',row='Survived')  
g.map(plt.hist, 'Age', alpha=0.8)  
plt.show()  




train['Sex'] = train.Sex.map({'male':1,'female':0})
test['Sex'] = test.Sex.map({'male':1,'female':0})
train_test = pd.concat([train,test])
train_test_missing_age = train_test[:]




age_data = train_test.drop(['PassengerId','Survived'],axis=1)
age_train = age_data[age_data.Age.notnull()].drop('Age',axis=1)
age_train_label = age_data.loc[age_data.Age.notnull(),'Age']
age_test_1 = train.drop(['PassengerId','Survived'],axis=1)[train.drop(['PassengerId','Survived'],axis=1).Age.isnull()].drop('Age',axis=1)
age_test_2 = test.drop(['PassengerId'],axis=1)[test.drop(['PassengerId'],axis=1).Age.isnull()].drop('Age',axis=1)




gb_regressor = GradientBoostingRegressor(loss='ls', criterion='mse',                                          learning_rate= 0.05, n_estimators=800,                                          max_depth=3, min_samples_leaf=2,                                          subsample=1, max_features=1, verbose=0)
gb_regressor.fit(age_train,age_train_label)




age_train_hat = gb_regressor.predict(age_train)
plt.scatter(age_train_label, age_train_hat, marker='+')
plt.plot([-1,80],[-1,80])
plt.xlabel('age_true')
plt.ylabel('age_hat')
plt.show()




missing_age_intrain = gb_regressor.predict(age_test_1)
missing_age_intest = gb_regressor.predict(age_test_2)
train.loc[train.Age.isnull(),'Age'] = missing_age_intrain.astype(int)
test.loc[test.Age.isnull(),'Age'] = missing_age_intest.astype(int)
train_test = pd.concat([train,test])




plt.hist(train_test.Age,bins=20, label='full_age')
plt.hist(train_test_missing_age.loc[train_test_missing_age.Age.notnull(), 'Age'],bins=20,alpha=0.6, label='missing_age')
plt.legend()
plt.show()




train['IsElderly'] = train.Age.apply(lambda x: 1 if x>=60 else 0)
test['IsElderly'] = test.Age.apply(lambda x: 1 if x>=60 else 0)
train['IsChildren'] = train.Age.apply(lambda x: 1 if x<=14 else 0)
test['IsChildren'] = test.Age.apply(lambda x: 1 if x<=14 else 0)




train.groupby('IsElderly').mean()['Survived']




X_train = train.drop(['PassengerId','Survived'],axis=1)
y_train = train.Survived
X_test = test.drop(['PassengerId'],axis=1)

xgb_clf = XGBClassifier(learning_rate=0.05, n_estimators=1000,
                        max_depth=3, min_child_weight=8, 
                        subsample=1, colsample_bytree=0.7,
                        reg_alpha=0, reg_lambda=1)
xgb_clf.fit(X_train, y_train)

pd.Series(xgb_clf.booster().get_fscore()).sort_values(ascending=False).plot(kind='bar')
plt.xlabel('feature_importance')
plt.show()

