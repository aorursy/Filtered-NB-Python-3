#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()




#Have a look at the data you have
df = pd.read_csv("../input/survey.csv")
list(df)




df.head()




#Use labelencoder to replace all categorical information with ordinal information where
#you don't care about the order
df.family_history = le.fit_transform(df.family_history) 
df.mental_health_consequence = le.fit_transform(df.mental_health_consequence)
df.phys_health_consequence = le.fit_transform(df.phys_health_consequence)
df.coworkers = le.fit_transform(df.coworkers)
df.supervisor = le.fit_transform(df.supervisor)
df.mental_health_interview = le.fit_transform(df.mental_health_interview)
df.phys_health_interview = le.fit_transform(df.phys_health_interview)
df.mental_vs_physical = le.fit_transform(df.mental_vs_physical)
df.obs_consequence = le.fit_transform(df.obs_consequence)
df.remote_work = le.fit_transform(df.remote_work)
df.tech_company = le.fit_transform(df.tech_company)
df.benefits = le.fit_transform(df.benefits)
df.care_options = le.fit_transform(df.care_options)
df.wellness_program = le.fit_transform(df.wellness_program)
df.seek_help = le.fit_transform(df.seek_help)
df.anonymity = le.fit_transform(df.anonymity)




df.loc[df['work_interfere'].isnull(),['work_interfere']]=0 # replace all NaNs with zero




#dealing with nulls
df['self_employed'].fillna('Don\'t know',inplace=True)
df.self_employed = le.fit_transform(df.self_employed)

#another way to deal with nulls
# Now change comments column to flag whether or not respondent made additional comments
df.loc[df['comments'].isnull(),['comments']]=0 # replace all no comments with zero
df.loc[df['comments']!=0,['comments']]=1 # replace all comments with a flag 1




#Preserve Order in some of the features
df['leave'].replace(['Very easy', 'Somewhat easy', "Don\'t know", 'Somewhat difficult', 'Very difficult'], 
                     [1, 2, 3, 4, 5],inplace=True) 
df['work_interfere'].replace(['Never','Rarely','Sometimes','Often'],[1,2,3,4],inplace=True)
#df.loc[df['work_interfere'].isnull(),['work_interfere']]=0 # replace all no comments with zero

#From assessing the unique ways in which gender was described above, the following script replaces gender on
#a -2 to 2 scale:
#-2:male
#-1:identifies male
#0:gender not available
#1:identifies female
#2: female.

#note that order of operations matters here, particularly for the -1 assignments that must be done before the
#male -2 assignment is done

df.loc[df['Gender'].str.contains('F|w', case=False,na=False),'Gender']=2
df.loc[df['Gender'].str.contains('queer/she',case=False,na=False),'Gender']=1
df.loc[df['Gender'].str.contains('male leaning',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('something kinda male',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('ish',case=False,na=False),'Gender']=-1
df.loc[df['Gender'].str.contains('m',case=False,na=False),'Gender']=-2
df.loc[df['Gender'].str.contains('',na=False),'Gender']=0




#preserve order in company size
df.loc[df['no_employees']=='1-5',['no_employees']]=1
df.loc[df['no_employees']=='6-25',['no_employees']]=2
df.loc[df['no_employees']=='26-100',['no_employees']]=3
df.loc[df['no_employees']=='100-500',['no_employees']]=4
df.loc[df['no_employees']=='500-1000',['no_employees']]=5
df.loc[df['no_employees']=='More than 1000',['no_employees']]=6




# Feature selection
drop_elements = ['Timestamp','Country','state','work_interfere']#work interfere goes because by defnition, if it inteferes with your work, then you definitely have a mental health issue
df = df.drop(drop_elements, axis = 1)




X = df.drop(['treatment'],axis=1)
y = df['treatment']
y = le.fit_transform(y) # yes:1 no:0




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=0.20, random_state=1)




from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

            ('pca', PCA(n_components=2)),
            ('clf', LogisticRegression(penalty='l2', C = 10000, random_state=1))])
#although l2 regularisation specified, this is the default set anyway. C parameter uses a default of 1




from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr, 
                         X=X_train, 
                         y=y_train, 
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('mean CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))




get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve




train_sizes, train_scores, cv_scores =                learning_curve(estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                train_sizes=np.linspace(0.1, 1.0, 10), 
                cv=10,
                n_jobs=1)
    
#The combination of train_sizes and cv set up the incremements to your data set. 
#For instance cv=10 divides data into 10 stratified folds 
#1/10 is the cross validation set
#9/10 is the training set
# That 9/10 is further divided into increasing train_sizes as determined by linspace
# see cell below for the numbers

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
cv_mean = np.mean(cv_scores, axis=1)
cv_std = np.std(cv_scores, axis=1)

plt.plot(train_sizes, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes, 
                 train_mean + train_std,
                 train_mean - train_std, 
                 alpha=0.15, color='blue')

plt.plot(train_sizes, cv_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(train_sizes, 
                 cv_mean + cv_std,
                 cv_mean - cv_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.4, 1.0])
plt.tight_layout()
# plt.savefig('./figures/learning_curve.png', dpi=300)
plt.show()




#Now change the logistic regression to no PCA to counter high bias
            #('pca', PCA(n_components=2)),
            ('clf', LogisticRegression(penalty='l2', random_state=1))])

train_sizes, train_scores, cv_scores =                learning_curve(estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                train_sizes=np.linspace(0.1, 1.0, 10), 
                cv=10,
                n_jobs=1)
    
#The combination of train_sizes and cv set up the incremements to your data set. 
#For instance cv=10 divides data into 10 stratified folds 
#1/10 is the cross validation set
#9/10 is the training set
# That 9/10 is further divided into increasing train_sizes as determined by linspace
# see cell below for the numbers

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
cv_mean = np.mean(cv_scores, axis=1)
cv_std = np.std(cv_scores, axis=1)

plt.plot(train_sizes, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes, 
                 train_mean + train_std,
                 train_mean - train_std, 
                 alpha=0.15, color='blue')

plt.plot(train_sizes, cv_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(train_sizes, 
                 cv_mean + cv_std,
                 cv_mean - cv_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.4, 1.0])
plt.tight_layout()
# plt.savefig('./figures/learning_curve.png', dpi=300)
plt.show()




scores = cross_val_score(estimator=pipe_lr, 
                         X=X_train, 
                         y=y_train, 
                         cv=10,
                         n_jobs=1)
print('mean CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))




from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                param_name='clf__C', 
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.6, 0.8])
plt.tight_layout()
# plt.savefig('./figures/validation_curve.png', dpi=300)
plt.show()




from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

            ('clf', SVC(random_state=1))])

param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range, 
               'clf__kernel': ['linear']},
                 {'clf__C': param_range, 
                  'clf__gamma': param_range, 
                  'clf__kernel': ['rbf']}]

# this bit is the inner loop
gs = GridSearchCV(estimator=pipe_svc, 
                            param_grid=param_grid, #this bit does the grid search of the parameter space i.e. linear/rbf and parameter tuning
                            scoring='accuracy', 
                            cv=2,
                            n_jobs=-1)

# Note: Optionally, you could use cv=2 
# in the GridSearchCV above to produce
# the 5 x 2 nested CV that is shown in the figure.

#gs = gs.fit(X_train, y_train)
#print(gs.best_score_) #whilst these numbers are interesting, they are not the outer loop cross-validation as
                      #below so will not be quoted as the training accuracy. 
#print(gs.best_params_)#

# this bit is the outer loop
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)




param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__C': param_range}]

#this bit is the inner loop
gs = GridSearchCV(estimator=pipe_lr, 
                            param_grid=param_grid, #this bit does the grid search of the parameter space i.e. linear/rbf and parameter tuning
                            scoring='accuracy', 
                            cv=2,
                            n_jobs=-1)

# Note: Optionally, you could use cv=2 
# in the GridSearchCV above to produce
# the 5 x 2 nested CV that is shown in the figure.

# this bit is the outer loop
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)




from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), 
                            param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], 
                            scoring='accuracy', 
                            cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)




from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
            ('clf', SVC(random_state=1))])

param_range = [0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'clf__C': param_range, 
               'clf__kernel': ['linear']}]#,
                 #{'clf__C': param_range, 
                  #'clf__gamma': param_range, 
                  #'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                            param_grid=param_grid, #this bit does the grid search of the parameter space i.e. linear/rbf and parameter tuning
                            scoring='accuracy', 
                            cv=10,
                            n_jobs=-1)

gs = gs.fit(X_train, y_train)
print(gs.best_score_) #whilst these numbers are interesting, they are not the outer loop cross-validation as
                      #below so will not be quoted as the training accuracy. 
print(gs.best_params_)#

y_true, y_pred = y_test, gs.predict(X_test)

#clf = gs.best_estimator_
#clf = gs.best_params_
#clf.fit(X_train, y_train)
#print('Test accuracy: %.3f' % clf.score(X_test, y_test))




from sklearn.metrics import classification_report




stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)




#svc = SVC(kernel='linear',C=0.1).fit(X_train_std, y_train)
clf = LogisticRegression(C=0.01).fit(X_train_std,y_train)
y_true, y_pred = y_test, clf.predict(X_test_std)
print(classification_report(y_true, y_pred))




order = np.argsort(abs(clf.coef_))
lab = clf.coef_.ravel()
lab = lab[order].T.ravel()
feature_names = X_train.columns[order].ravel()




plt.figure(figsize=(13, 5))
plt.bar(np.arange(len(X_train.columns)),lab)
plt.xticks(np.arange(1+len(X_train.columns)),feature_names, rotation=60, ha='right')
plt.show()




print(le.inverse_transform(df.family_history)[:5])
print(df.family_history.head())




df.care_options.head()




df_o = pd.read_csv('../input/survey.csv')




df_o.care_options.head()




df_o.Gender.head()




df.Gender.head()




df_o.benefits.head()




df.benefits.head()

