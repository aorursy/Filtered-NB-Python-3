#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




data=pd.read_csv('../input/train.csv')
data.head()




data.describe()




def combined_data():
    train=pd.read_csv('../input/train.csv')
    test=pd.read_csv('../input/test.csv')
    target=train.Survived
    train.drop(['Survived'],axis=1,inplace=True)
    combined=train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index','PassengerId'],inplace=True,axis=1)
    return combined




combined=combined_data()
combined.drop(['Name','Cabin'],axis=1,inplace=True)
combined.head()




combined.iloc[891:].isnull().sum()




combined['Fare'] = combined['Fare'].fillna(combined['Fare'].median())
combined.iloc[891:].isnull().sum()




grouped_data=combined.iloc[:891].groupby(['Sex','Pclass'])
grouped_data_median=grouped_data.median()
grouped_data_median=grouped_data_median.reset_index()[['Sex','Pclass','Age']]
combined.head()




grouped_data_median
combined.isnull().sum()




def fill_age(row):
    condition = (
        (grouped_data_median['Sex'] == row['Sex']) & 
        (grouped_data_median['Pclass'] == row['Pclass'])
    ) 
    return grouped_data_median[condition]['Age'].values[0]

def process_age():
    global combined
    # a function that fills the missing values of the Age variable
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
   # status('age')
    return combined
combined.isnull().sum()




combined=process_age()
combined['Fare'] = pd.qcut(combined['Fare'], 4,labels=["1","2","3","4"])
combined=pd.concat([combined,pd.get_dummies(combined['Fare'],prefix='Fare_')],axis=1)
combined['Age'] = pd.qcut(combined['Age'], 5,labels=["1","2","3","4","5"])
combined=pd.concat([combined,pd.get_dummies(combined['Age'],prefix='Age_')],axis=1)
combined.drop(['Fare','Age'],axis=1,inplace=True)
#combined=pd.rename(columns={'(-0.001, 7.896]':'f1','(7.896, 14.454]':'f2','(14.454, 31.275]':'f3','(31.275, 512.329]':'f4',
                          # '(0.169, 21.0]':'a1','(21.0, 25.0]':'a2','(25.0, 30.0]':'a3','(30.0, 40.0]':'a4','(40.0, 80.0]':'a5'})
combined.head()




def sex_process():
    global combined
    combined['Sex']=combined['Sex'].map({'male':1,'female':0})
    return combined
combined=sex_process()
combined.head()




def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")
    
    # adding dummy variable
    combined = pd.concat([combined, pclass_dummies],axis=1)
    
    # removing "Pclass"
    combined.drop('Pclass',axis=1,inplace=True)
    
    #status('Pclass')
    return combined
#combined=process_pclass()




combined.drop('Ticket',inplace=True,axis=1)
#combined.drop('Ticket',inplace=True,axis=1)
#combined['SibSp'].value_counts().plot(kind='bar')
combined.head()




def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    combined.Embarked.fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    #status('embarked')
    return combined
combined = process_embarked()









combined.head()
#combined.drop(['Embarked'],inplace=True,axis=1)
combined.head()




#combined=combined.drop('Embarked',axis=1)




from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier 




def compute_score(clf,X,y,scoring='accuracy'):
    xval=cross_val_score(clf,X,y,cv=5,scoring=scoring)
    return np.mean(xval)




def recover_train_test_targets():
    targets=pd.read_csv('../input/train.csv',usecols=['Survived'])['Survived'].values
    targets=targets[0:693]
    train=combined.iloc[:693]
    test=combined.iloc[891:]
    return train ,test, targets




train,test,targets = recover_train_test_targets()
#train.shape
#clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
#clf = clf.fit(train, targets)




#train['Embarked']=train['Embarked'].fillna('S')









#train.head()




#clf=RandomForestClassifier(n_estimators=50,max_features='sqrt')
#clf=clf.fit(train,targets)
#param_grid = [{'min_child_weight': np.arange(0.1, 10.1, 0.1)}]
#clfd=GridSearchCV(XGBClassifier(), param_grid, cv=10, scoring= 'f1',iid=True)
    # model.fit(xtr, ytr)
clfd=XGBClassifier()
clfd.fit(train,targets)




features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clfd.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)




features.plot(kind='barh', figsize=(25, 25))




output = clfd.predict(test)
df_output = pd.DataFrame()
aux = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
#df_output[['PassengerId','Survived']].to_csv('../input/gender_submission.csv', index=False)
submission = pd.DataFrame({
        "PassengerId": df_output["PassengerId"],
        "Survived": df_output["Survived"]
    })
submission.to_csv('titanic.csv', index=False)
#df_output.to_csv('../input/gender_submission.csv',index=False)
#score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')




name=pd.read_csv('titanic.csv')
name.head(50)
















