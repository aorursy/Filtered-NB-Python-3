#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (12,12)




data = pd.read_csv('../input/train.csv')
data.head()




data.describe()




data.describe(include=['O'])




# map 'Sex' (string) to 'gender' (int)
data['Sex'] = data['Sex'].astype('category')
data['Embarked'] = data['Embarked'].astype('category')

# impute Age based on pclass and gender
lookup = np.zeros([2,3])
sexdict = {0:'male', 1:'female'}
for s in [0,1]:
    thesex = sexdict[s]
    for pc in range(0,3):
        lookup[s,pc] = data[ (data.Sex == thesex) & (data.Pclass == pc +1) ]['Age'].median()
# fill dataframe based on lookup table
for s in [0,1]:
    thesex = sexdict[s]
    for pc in range(0,3):
        data.loc[(data.Sex ==thesex ) & ( data.Pclass == pc +1) & (data.Age.isnull()),'Age'] = lookup[s,pc]
# sub siblings spouses and parents

data['Family'] = data['SibSp'] + data['Parch']
lookup        




data.describe()




# pclass/gender survival rates
data.groupby(['Pclass','Sex'])['Survived'].agg(['mean','count'])




data.groupby(['Pclass','Embarked'])['Survived'].agg(['mean','count'])




data.groupby(['Pclass','Sex','Survived']).size()




# plot Age distribution for survived/died
facet = sns.FacetGrid(data,hue='Survived',size=7)
facet.map(sns.distplot,'Age',hist=False,kde_kws={"shade": True})
facet.add_legend()

#f,axes = plt.subplots(2,2)
#sns.distplot(data['Age'],ax=axes[0,0])




facet = sns.FacetGrid(data,hue='Survived',size=7,row='Pclass',col='Sex')
facet.map(sns.distplot,'Age',hist=True,kde=False,kde_kws={"shade": True})
facet.add_legend()




# This looks kinda nice, but isn't the best way to visualise this. 
# family is an integer, and the valid values don't always align with the hex grids.
#with sns.axes_style("white"):
# specify extent=[a,b,c,d] to have all plots use the same grid
#    g = sns.FacetGrid(data,row='Survived',col='Pclass',size=7)
#    g.map(plt.hexbin,'Age','Family',linewidths=0.0,gridsize=10,cmap=sns.cubehelix_palette(20, start=0, rot=0,as_cmap=True))

#    g.add_legend()

#g = sns.FacetGrid(data,row='Survived',col='Pclass',size=7,hue='Survived')
#g.map(sns.kdeplot,'Age','Family')
with sns.axes_style("white"):
    g = sns.FacetGrid(data,row='Survived',col='Pclass',size=7)
    g.map(sns.kdeplot,'Age','Family',cmap=sns.cubehelix_palette(20, start=0, rot=0,as_cmap=True))

    g.add_legend()




from sklearn.model_selection import train_test_split

# create dummy variables for categoricals.
# Pclass should be included here also, as class number isn't really quantitative
MLdata=pd.get_dummies(data=data,columns=['Sex','Embarked','Pclass'])


# drop some columns
MLdata = MLdata.drop(['PassengerId','Ticket','Cabin','Name','Family'],axis=1)
#MLdata.head()

# subset data, seperate predictors and regressor
#PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
Y = MLdata['Survived']
X=MLdata.drop('Survived',axis=1)

X_train, X_val,Y_train,Y_val =     train_test_split(X,Y,test_size=0.2,random_state=96)
X_train.head()




# import random forest and cross validation grid search
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# set the seed for reproducability 
import random
random.seed(42)

# tune our forest
# set up a parameter grid. Will vary number of trees and number of predictors to consider at each split
parameterGrid = {
    'max_features': ['log2','sqrt',3],
    'n_estimators': [100,200,300,500]
}

# setup the grid, fit it to the training subset
cvgrid = GridSearchCV(RandomForestClassifier(),param_grid=parameterGrid,cv=10,n_jobs=2,refit=False)
cvgrid.fit(X_train,Y_train)

# print results of the gridsearch, based on ranking
ranking = cvgrid.cv_results_['rank_test_score'] 
# list of Nones to hold output
strs = [None for i in ranking ]

#loop over grid results, insert performance in appropriate place in list
for index,rank in enumerate(ranking):
    mean = cvgrid.cv_results_['mean_test_score'][index]
    stddev = cvgrid.cv_results_['std_test_score'][index]
    # format the results
    thestr = '{r:2}- max_features: {mf:10}, n_estimators: {ne:4}, {m:5.4g} +/- {sd:5.4g}'.format(
        r=rank,
        mf=cvgrid.cv_results_['params'][index]['max_features'],
        ne=cvgrid.cv_results_['params'][index]['n_estimators'],
        m=mean,
        sd=stddev
    )
    
    # if there are ties, adjust the ranking index
    # (if two parameter combinations score the same,
    # the second is ordered as if it were ranked one worse)
    checktie = rank -1 
    while strs[checktie]:
        checktie +=1
    strs[checktie] = thestr

# print stuff    
for i in strs:
    print(i)




# fit a forest to the training subset, based on the parameters found from the gridsearch
nest = cvgrid.best_params_['n_estimators']
maxf = cvgrid.best_params_['max_features']
clf_randForest = RandomForestClassifier(n_estimators=nest,max_features=maxf,oob_score=True)

# train the forest
clf_randForest.fit(X_train,Y_train)

# check the oob score
print('Out-of-Bag accuracy = {a:5.4g}'.format(a=clf_randForest.oob_score_))

# score on validation set
val_score = clf_randForest.score(X_val,Y_val)

print('Accuracy on validation set = {a:5.4g}'.format(a=val_score))




# ROC curve, auc
from sklearn import metrics
# use our validation set to get the ROC curve
# could also do this using OOB samples, oob_decision_fcuntion_
val_survived_probs = clf_randForest.predict_proba(X_val)[:,1]
# Averages class probabilities over all trees in the ensemble.
# The class probability for a given tree is the fraction of observations belonging to that class in the leaf node.
fpr, tpr, thresholds = metrics.roc_curve(Y_val,val_survived_probs,pos_label=1)
auc = metrics.roc_auc_score(Y_val,val_survived_probs)

plt.plot(fpr,tpr,'r-',label='Random Forest ROC: auc = {a:4.3g}'.format(a=auc))
plt.plot([0,1],[0,1],linestyle='--',color=sns.xkcd_rgb['royal purple'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve - random forest')
plt.legend(loc='lower right')
#plt.text(0.7,0.4,'auc = {a:4.3g}'.format(a=auc))

