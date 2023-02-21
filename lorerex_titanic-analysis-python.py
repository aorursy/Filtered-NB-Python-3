#!/usr/bin/env python
# coding: utf-8



#%matplotlib inline

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline

sns.set(style='white', context='notebook', palette='deep')




train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_plus_test = pd.concat([train.drop('Survived',1),test])




print(train.head())




train.describe()




print(train.isnull().sum(), '\n-------------')
print(test.isnull().sum())




surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]
surv_col = "green"
nosurv_col = "red"

print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"      %(len(surv), 1.*len(surv)/len(train)*100.0,        len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train)))




warnings.filterwarnings(action="ignore")
plt.figure(figsize=[12,10])
plt.subplot(331)
sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,
            axlabel='Age')
plt.subplot(332)
sns.barplot('Sex', 'Survived', data=train)
plt.subplot(333)
sns.barplot('Pclass', 'Survived', data=train)
plt.subplot(334)
sns.barplot('Embarked', 'Survived', data=train)
plt.subplot(335)
sns.barplot('SibSp', 'Survived', data=train)
plt.subplot(336)
sns.barplot('Parch', 'Survived', data=train)
plt.subplot(337)
sns.distplot(surv['Fare'].dropna().values+1, kde=False, color=surv_col)
sns.distplot(nosurv['Fare'].dropna().values+1, kde=False, color=nosurv_col,axlabel='Fare')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

print("Median age survivors: %.1f, Median age non-survivers: %.1f"      %(np.median(surv['Age'].dropna()), np.median(nosurv['Age'].dropna())))




# filling age Nans

print('Oldest Passenger was of:',train_plus_test['Age'].max(),'Years')
print('Youngest Passenger was of:',train_plus_test['Age'].min(),'Years')
print('Average Age on the ship:',train_plus_test['Age'].mean(),'Years')




# two new features:
# Title (Mister, Miss, etc...) should be pretty relevant, even though it's likely correlated with age and ticket class
train_plus_test['Title'] = train_plus_test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
# Could an unknown age be relevant in surviving? Probably not, but let's give it a try. 
train_plus_test['Age_known'] = train_plus_test['Age'].isnull() == False




pd.crosstab(train_plus_test.Title,train_plus_test.Sex).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the Sex




# let's fix spelling errors and group titles into few categories
train_plus_test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)




train_plus_test.groupby('Title')['Age'].mean() #lets check the average age by Initials




## Replacing the NaN Values with the mean ages. 
train_plus_test.loc[(train_plus_test.Age.isnull())&(train_plus_test.Title=='Mr'),'Age']=33
train_plus_test.loc[(train_plus_test.Age.isnull())&(train_plus_test.Title=='Mrs'),'Age']=37
train_plus_test.loc[(train_plus_test.Age.isnull())&(train_plus_test.Title=='Master'),'Age']=5
train_plus_test.loc[(train_plus_test.Age.isnull())&(train_plus_test.Title=='Miss'),'Age']=22
train_plus_test.loc[(train_plus_test.Age.isnull())&(train_plus_test.Title=='Other'),'Age']=45




train_plus_test.loc[(train_plus_test.Fare.isnull())]




row_to_fix = train_plus_test.loc[(train_plus_test.Fare.isnull())]
# let's temporary remove the problematic passenger
train_plus_test = train_plus_test[train_plus_test.PassengerId != 1044]




f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(train_plus_test[train_plus_test['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(train_plus_test[train_plus_test['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(train_plus_test[train_plus_test['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()




f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(train_plus_test[train_plus_test['Age'] > 50].Fare,ax=ax[0])
ax[0].set_title('Fares in Age > 50')
sns.distplot(train_plus_test[(train_plus_test['Age'] <= 50 ) &  (train_plus_test['Age'] >= 25)].Fare,ax=ax[1])
ax[1].set_title('Fares in Age < 50 and Age > 25')
sns.distplot(train_plus_test[train_plus_test['Age'] < 25].Fare,ax=ax[2])
ax[2].set_title('Fares in Age < 25')
plt.show()




print("Average fare for Pclass 3 and Age > 50")
train_plus_test.loc[(train_plus_test.Pclass == 3 ) & (train_plus_test.Age > 50 )]['Fare'].mean()




# Assigning the average fare to Mr. Thomas 
row_to_fix['Fare'] = 7.7
print(row_to_fix)
train_plus_test = train_plus_test.append(row_to_fix)
print(train_plus_test.iloc[-1])




print(train_plus_test.tail())
train_plus_test = train_plus_test.sort_values('PassengerId')
print(train_plus_test.tail())




train_plus_test.loc[(train_plus_test.Embarked.isnull())]




f, (ax1,ax2)=plt.subplots(ncols=2, sharey = True, figsize=(10,5))
sns.countplot('Embarked',data=train_plus_test,ax=ax1)
ax1.set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Pclass',data=train_plus_test,ax=ax2)
ax2.set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()




# Since the two passengers are first class they likely come from S. The other possibility would be C.
# Moreover S is the more likely class anyway.
train_plus_test.Embarked[train_plus_test.PassengerId == 62] = 'S'
train_plus_test.Embarked[train_plus_test.PassengerId == 830] = 'S'




survived = train['Survived']




# for many classifiers it turns out to be efficient to discretize continuous features 
#(actually continuous ones are  not bad either).

#I choose the numbers by hand, 
# assuring that every class has a fair number of instances.
# class 0 is basically childrens, while class 1 are still-not-adult people (21 was adulthood in
# that period)
# class 2 and 3 is the typical age in which men must give priority to elders, children
# and women in case of emergency.

train_plus_test['Age_band']=0
train_plus_test.loc[train_plus_test['Age']<=15,'Age_band']=0
train_plus_test.loc[(train_plus_test['Age']>=16)&(train_plus_test['Age']<=21),'Age_band']=1
train_plus_test.loc[(train_plus_test['Age']>=22)&(train_plus_test['Age']<=28),'Age_band']=2
train_plus_test.loc[(train_plus_test['Age']>=29)&(train_plus_test['Age']<=36),'Age_band']=3
train_plus_test.loc[(train_plus_test['Age']>=37)&(train_plus_test['Age']<=50),'Age_band']=4
train_plus_test.loc[train_plus_test['Age']>=51,'Age_band']=5
train_plus_test.head(2)

f, (ax1)=plt.subplots(ncols=1, sharey = True, figsize=(10,5))
sns.countplot('Age_band',data=train_plus_test,ax=ax1)
ax1.set_title('No. Of Passengers Boarded')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show() 




f, (ax1)=plt.subplots(ncols=1, sharey = True, figsize=(10,5))
sns.distplot(train_plus_test.Fare,ax=ax1)
ax1.set_title('No. Of Passengers Boarded')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.xlim(0, 60)
plt.show()
f, (ax2)=plt.subplots(ncols=1, sharey = True, figsize=(10,5))
sns.countplot('Pclass',data=train_plus_test,ax=ax2)
ax2.set_title('No. Of Passengers Boarded')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()




# let's discretize the fare feature. Here class 1 is the price for the average ticket, 
# which is driven by the 3rd class ticket. Class 0 is a bit puzzling, maybe people who got it 
# with some discount. Again, I build the classes in such a way to have equally distributed 
# instances in each class.

train_plus_test['Fare_range']=0
train_plus_test.loc[train_plus_test['Fare']<=6,'Fare_range']=0
train_plus_test.loc[(train_plus_test['Fare']>=7)&(train_plus_test['Fare']<=8),'Fare_range']=1 
train_plus_test.loc[(train_plus_test['Fare']>=9)&(train_plus_test['Fare']<=15),'Fare_range']=2
train_plus_test.loc[(train_plus_test['Fare']>=16)&(train_plus_test['Fare']<=26),'Fare_range']=3
train_plus_test.loc[(train_plus_test['Fare']>=27)&(train_plus_test['Fare']<=36),'Fare_range']=4
train_plus_test.loc[(train_plus_test['Fare']>=37)&(train_plus_test['Fare']<=67),'Fare_range']=5
train_plus_test.loc[train_plus_test['Fare']>=68,'Fare_range']=6
train_plus_test.head(2)

f, (ax1)=plt.subplots(ncols=1, sharey = True, figsize=(10,5))
sns.countplot('Fare_range',data=train_plus_test,ax=ax1)
ax1.set_title('No. Of Passengers Boarded')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()




# Perhaps relevant?
train_plus_test['Cabin_known'] = train_plus_test['Cabin'].isnull() == False
# Being Alone should change a lot, no family members / friends to search in case of emergency! 
# But no help from them either...
train_plus_test['Alone']  = (train_plus_test['SibSp'] + train_plus_test['Parch']) == 0
# very likely a big problem in order to survive. Especially for the father!
train_plus_test['Large_Family'] = (train_plus_test['SibSp']>2) | (train_plus_test['Parch']>3)
# correlated with many of the above, but let's see.
train_plus_test['Shared_ticket'] = np.where(train_plus_test.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)

print(train.shape)
print(test.shape)

test = train_plus_test.iloc[len(train):]
train = train_plus_test.iloc[:len(train)]
train['Survived'] = survived




# In principle one may try some more advanced insights for the features below. In principle!
test_Ids = test['PassengerId']
columns_to_drop =  ['Name', 'Ticket', 'Cabin', 'PassengerId'
                    ,'Age', 'Fare'
                   ]
train = train.drop(columns_to_drop, axis = 1 )
test = test.drop( columns_to_drop, axis = 1 )




def one_hot( train_test, column_name):
    # here train_test is a list [train, test]
    train_test[column_name] = train_test[column_name].astype("category")
    # transform different features in different integers. 
    train_test[column_name].cat.categories = list(range(len(train_test[column_name].unique())))
    train_test[column_name] = train_test[column_name].astype("int")
    
    return train_test




survived = train['Survived']

features_to_one_hot = ["Sex", "Embarked", "Title","Large_Family", "Alone"
                       ,"Cabin_known","Age_known"
                      ]
for x in [train, test]: 
    for feature in features_to_one_hot:
        x =  one_hot( x, feature)

print(train.head())




ax = plt.subplots( figsize =( 12 , 10 ) )
foo = sns.heatmap(train.corr(), vmax=1.0, square=True, annot=True)




# train-validation split  (validation should not be confused with the final test set)
# stratify will help testing on a realistic distribution
# apparently the final accuracy is pretty sensitive on random_state. I guess that this is due to 
# the very small size of the dataset.
X_train, X_validation  = train_test_split(train, test_size=0.2, 
                                          random_state = 3,
                                                                stratify = 
                                                                #  None)
                                                                  survived)




y_train = X_train['Survived']
y_validation = X_validation['Survived']

X_train =  X_train.drop('Survived', axis = 1 )
X_validation =  X_validation.drop('Survived', axis = 1 )




print(X_train.shape)
print(X_train.columns)
print(X_train.columns.shape[0])
print("Total sample size = %i; training sample size = %i, testing sample size = %i"     %(train.shape[0],X_train.shape[0],X_validation.shape[0]))




def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confusion_matrix=True):
    from sklearn import metrics
    y_pred = clf.predict(X)

    print("##########################", '\n')
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)), "\n")
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred), "\n")
    if show_confusion_matrix:
        print("Confusion matrix")
    print(metrics.confusion_matrix(y, y_pred), "\n")
print("##########################", '\n')




# N_features to use. In principle one can gridsearch on it, but here we don't and use all the features. 
N_FEATURES_OPTIONS = [X_train.columns.shape[0]]  

# parameters to grid search
learning_rates = [0.05,0.075,0.1]
max_depths = [5, 7,  None]
n_estimators = [10,100,1000]
class_weights = ['balanced', None]
n_est = 500
base_scores= [0.5, 0.6, 0.7]
criterions = ['gini', 'entropy']
min_samples_leaf = [3,5,7]
min_samples_split = [5, 7, 8]
Cs  = [1.0, 1.6, 1.75, 2.0]
gammas = [0.05,0.1, 'auto']
ks = [8, 10, 12]
subsamples = [0.5,0.75,1.0]


classifier0  = xgb.XGBClassifier(max_depth=7,n_estimators=n_est,learning_rate=0.8)
param_grid0 = [{
        'featureSelection__k': N_FEATURES_OPTIONS,
              'classify__learning_rate': learning_rates,
            'classify__base_score': base_scores,
              }]

classifier1  = RandomForestClassifier(n_estimators=n_est,criterion='entropy',max_depth=None,min_samples_leaf=3,min_samples_split=8
                                      ,class_weight = None)
param_grid1 = [{
        'featureSelection__k': N_FEATURES_OPTIONS,
          #    'classify__max_depth': max_depths,
    #'classify__criterion': criterions,
      #  'classify__min_samples_leaf': min_samples_leaf,
      #  'classify__min_samples_split': min_samples_split,
     #   'classify__class_weight': class_weights
              }]

classifier2 = ExtraTreesClassifier(n_estimators=n_est,criterion='entropy',max_depth=5,min_samples_leaf=5,min_samples_split=7)
param_grid2 = [{
       # 'featureSelection__k': N_FEATURES_OPTIONS,
             # 'classify__max_depth': max_depths,
    #'classify__criterion': criterions,
       # 'classify__min_samples_leaf': min_samples_leaf,
    #    'classify__min_samples_split': min_samples_split,
   # 'classify__class_weight': class_weights
              }]

classifier3  = AdaBoostClassifier(n_estimators=n_est,learning_rate=0.1)
param_grid3 = [{
        'featureSelection__k': N_FEATURES_OPTIONS,
              'classify__learning_rate': learning_rates,
              }]

classifier4 = svm.SVC(class_weight=None, gamma= 'auto', C = 2.0)
param_grid4 = [{ 
        'featureSelection__k': N_FEATURES_OPTIONS,
              'classify__C': Cs,
         #   'classify__gamma': gammas,
    #'classify__class_weight': class_weights
              }]

classifier5 = KNeighborsClassifier(n_neighbors=10,weights='distance')
param_grid5 = [{
        'featureSelection__k': N_FEATURES_OPTIONS,
              'classify__n_neighbors': ks,
              }]

classifier6 = GradientBoostingClassifier( n_estimators=n_est,learning_rate=0.1)
param_grid6 = [{
        'featureSelection__k': N_FEATURES_OPTIONS,
        'classify__max_depth': max_depths,
        'classify__learning_rate': learning_rates,
         'classify__min_samples_leaf': min_samples_leaf,
        #'classify__min_samples_split': min_samples_split,
              }]


classifiers = [
    #classifier0,
    classifier1
   #            , classifier2, 
    #classifier3, 
    ,classifier4
               #, classifier5, classifier6
              ]
param_grids = [
    #param_grid0,
                 param_grid1
   #            , param_grid2
    #, param_grid3
    , param_grid4
               #, param_grid5, param_grid6
              ]

clf_names = [
            #'XGB',
             'RandomForest'
    #,'ExtraTrees' 
             #,'Ada'
             , 'SVC'
             #,'Knear', 'GradientBoost'
            ]




def train_and_validate(X_train, y_train, classifier, param_grid,  number_of_cross_validations = 10):
    
    feature_selector = SelectKBest(score_func= mutual_info_classif, k = 3)

    
    pipe = Pipeline([
    ('featureSelection', feature_selector),
    ('classify', classifier)
                ])

    print("grid searching ...")
    
    grid = GridSearchCV(pipe, cv=number_of_cross_validations, n_jobs=1, param_grid=param_grid)
    classifier = grid.fit(X_train, y_train)

    # print grid search results
    print("Best Estimator: ", grid.best_estimator_)
    print("Best Parameters: ", grid.best_params_)
    print("Best Score: ", grid.best_score_)
    
    
    # evaluation of the results
    measure_performance(X_train, y_train, classifier, show_accuracy=True, show_classification_report=True,
     show_confusion_matrix=True)

    # print the name of the selected columns
    finalFeatureIndices = grid.best_estimator_.named_steps["featureSelection"].get_support(indices=True)
    finalFeatureList = [X_train.columns.values.tolist()[i] for i in finalFeatureIndices]
    print("Selected Features: ", finalFeatureList)
    
    return classifier, finalFeatureList, grid

    




def validating(classifier, X_validation, y_validation, grid, finalFeatureList, n_cv = 5):
    
    scores = cross_val_score(classifier, X_validation, y_validation, cv=n_cv)
    print(scores)
    mean_score = np.mean(scores)
    print("Mean score = %.3f, Std deviation = %.3f"%(mean_score,np.std(scores)))
    
    score = classifier.score(X_validation, y_validation)
    print(score)
    
    #show feature importance
    if hasattr(grid.best_estimator_.named_steps['classify'], 'feature_importances_'):
        print(pd.DataFrame(list(zip(finalFeatureList, np.transpose(grid.best_estimator_.named_steps['classify'].feature_importances_)))).sort_values(1, ascending=False),'\n')
    
    print('\n', '################################', '\n')
    print('\n', '--------------------------------' '\n')
    print('\n', '################################', '\n')
    
    return mean_score, score 




get_ipython().run_cell_magic('time', '', "\nsummary = {}\n\nbest_classifier = None\nbest_mean_score = 0\n\nfor i in range(len(classifiers)):\n    \n    classifier =  classifiers[i]\n    param_grid = param_grids[i]\n    print(classifier,'\\n' ,param_grid)\n    \n    %time  classifier, finalFeatureList, grid = train_and_validate(X_train, y_train, classifier, param_grid)\n    \n    %time mean_score, score  = validating(classifier, X_validation, y_validation, grid, finalFeatureList)\n    \n    if (mean_score > best_mean_score):\n        best_mean_score = mean_score\n        best_classifier = classifier\n        \n    \n    summary[clf_names[i]] = [mean_score, score]")




print('\t  Mean Score, \t \t Score \n')
for name, scores in summary.items():
    print(name, scores, '\n')




X_test = test.loc[:,finalFeatureList]
print(best_classifier.best_estimator_.named_steps['classify'])
surv_pred = (best_classifier.best_estimator_.named_steps['classify']).predict(X_test)
print("Test Survivors: ", np.count_nonzero(surv_pred == 1))
print("Test Fatalities: ", np.count_nonzero(surv_pred == 0))




submit = pd.DataFrame({'PassengerId' : test_Ids,
                       'Survived': surv_pred.T})
submit.to_csv("submit.csv", index=False)




submit.head()




submit.tail()




submit.shape

