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




import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import pandas as pd
import random
import itertools
import seaborn as sns

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
import mpld3 as mpl

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from __future__ import print_function
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.metrics import classification_report,roc_curve, auc
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_selection import SelectKBest

from sklearn.preprocessing import StandardScaler

from sklearn import tree
from IPython.display import Image   




bc = pd.read_csv('../input/data.csv')

# remove last column: it is only garbage...
bc = bc.drop("Unnamed: 32", axis=1)

# NOTE: ["id","diagnosis"] dont need to be rescaled
# they correspond to indexes [0,1]  --> df.ix[start:end]
tmp = bc.copy() 
bc = bc.drop('id', axis=1)
tmp['diagnosis'] = bc['diagnosis'].apply(lambda diagnosis: 0 if diagnosis == "B" else 1)

# i want 'diagnosis' in another variable target
bc = bc.drop('diagnosis', axis=1)
target = tmp['diagnosis']

name = list(bc)
name.append('diagnosis')

### STANDARDIZATION ###
bc_std = (bc - bc.mean()) / (bc.std())
# values are now N(0,1)

### NORMALIZATION ###
bc_norm = (bc - bc.min()) / (bc.max() - bc.min())
# values are now in [0,1]

### LOG-NORMALIZATION ###
bc_log = bc.apply(np.log2)

# need to drop them cause there were -inf values!!! (log 0 -> -inf !!!)
bc_log = bc_log.drop(['concavity_mean','concave points_mean','concavity_se','concave points_se','concavity_worst','concave points_worst'], 1)




df = bc_std.copy()
df['diagnosis'] = target
g = sns.pairplot(df, hue = 'diagnosis', palette = 'Blues_d', kind="reg", vars=name[26:30])




df = bc_log.ix[:,['radius_mean', 'texture_mean', 'perimeter_mean','area_mean']]
df['diagnosis'] = target
mbc = pd.melt(df, "diagnosis", var_name="measurement")
fig, ax = plt.subplots(figsize=(12,5))
p = sns.violinplot(ax = ax, x="measurement", y="value", hue="diagnosis", split = True, data=mbc, inner = 'quartile', palette = 'Set1');
p.set_xticklabels(rotation = 0, labels = list(bc_std.columns));




corr = bc_std.corr() # .corr is used for find corelation
plt.figure(figsize=(12,12))
sns.heatmap(corr, cbar = True,  square = False, annot=False, fmt= '.2f',annot_kws={'size': 15},
           #xticklabels= name, 
            #yticklabels= features_mean,
           cmap= 'coolwarm') # for more on heatmap you can visit Link(http://seaborn.pydata.org/generated/seaborn.heatmap.html




# Feature Importance
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


X = bc_std.ix[:,0:30]
y = target

class_names = list(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(4),scoring='accuracy', verbose=0)
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

name = bc_std.columns
name = name[0:30] #remove diagnosis

#print(name.size)
print( 'selected attributes are:')

selectedFeatures = []

c=0
for i in np.arange(rfecv.support_.size):
    if rfecv.support_[i]==True :
        print('%f \t %s' % (rfecv.grid_scores_[i],name[i]))
        selectedFeatures.append(name[i])
        c=c+1

# Reduce X to the selected features.
XafterRFECV = rfecv.transform(X)

# convert np.array --> pandas.df
bc_std_sel = pd.DataFrame(XafterRFECV)

# add name to selected columns
bc_std_sel.columns = selectedFeatures




### PCA STD ###
from sklearn.decomposition import PCA

df = pd.DataFrame(data=bc_std_sel)
df['diagnosis'] = target

pca = PCA(n_components=10)
pca.fit(df)

cumExplainedVar = np.zeros(pca.n_components)

for i in range(len(pca.explained_variance_ratio_)):
    if i==0:
        cumExplainedVar[0]=pca.explained_variance_ratio_[0]
    else:
        cumExplainedVar[i] += cumExplainedVar[i-1] + pca.explained_variance_ratio_[i]

plt.plot(pca.explained_variance_ratio_, '-k', label='explained variance ratio')
plt.xlabel('Number principal components')
plt.xticks(np.arange(1,20))
plt.xlim(0,7)
plt.title('PCA')
plt.grid(True)
plt.hold
plt.plot(cumExplainedVar, '-r' , label='cumulative explained variance ratio')
plt.xlabel('Number principal components')
plt.xticks(np.arange(1,20))
plt.xlim(0,7)
plt.legend()
plt.show()




df = pd.DataFrame(data=bc_std_sel)
df['diagnosis'] = target
#first we need to map colors on labels
dfcolor = pd.DataFrame([[1,'red'],[0,'black']],columns=['diagnosis','Color'])
mergeddf = pd.merge(df,dfcolor)

pca = PCA(n_components=6)
df = df.drop('diagnosis', axis=1)
pca.fit(df)

X_pca = pca.fit_transform(df)

bc_pca = pd.DataFrame(X_pca)

#Then we do the graph

plt.scatter(X_pca[:,0],X_pca[:,1],color=mergeddf['Color'])
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import pylab

fig = pylab.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = X_pca[:,0]
sequence_containing_y_vals = X_pca[:,1]
sequence_containing_z_vals = X_pca[:,2]

random.shuffle(sequence_containing_x_vals)
random.shuffle(sequence_containing_y_vals)
random.shuffle(sequence_containing_z_vals)

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c=mergeddf['Color'])
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.legend()
pyplot.show()




# PCA after RFECV

bc_std_sel_pca = pd.DataFrame(data=bc_std_sel)
bc_std_sel_pca['diagnosis'] = target

pca = PCA(n_components=5)
pca.fit(bc_std_sel_pca)




### select dataset ###
X = bc_std_sel_pca
y = target
class_names = list(y.unique())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=14)
df = X.copy()
df['diagnosis']=target

traindf, testdf = train_test_split(df, test_size = 0.2)

predictor_var = name
outcome_var='diagnosis'

### GRIDSEARCH ###
scores = ['accuracy']

def run_model(model_type,model_short_name,tuning_parameters):
    for score in scores:
        print("# Tuning hyper-parameters for %s on scoring method: %s" % (model_short_name,score))
        print()

        clf = GridSearchCV(model_type, tuning_parameters, cv=5,
                           scoring='%s' % score)
        clf.fit(X_train, y_train)
             
        #print("Best parameters set found on development set:")
        print(clf.best_params_)
        print()
        #print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.5f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        #print("Detailed classification report:")
        print()
        #print("The model is trained on the full development set.")
        #print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        
        tp=0
        tn=0
        fp=0
        fn=0
        
        # convert pandas.series to np.array
        l = np.array(y_true)
           
        for i in range(1,len(y_true)):
            if(y_pred[i]==1 and l[i]==1):
                tp+=1              
            if(y_pred[i]==1 and l[i]==0):
                fp+=1       
            if(y_pred[i]==0 and l[i]==1):
                fn+=1
            if(y_pred[i]==0 and l[i]==0):
                tn+=1
                
        accuracy = (tp+tn) / (tp+fp+tn+fn)
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f = 2 * precision * recall / (precision + recall)
        
        print("accuracy\t%f \nrecall\t\t%f\nprecision\t%f\nf1\t\t%f\n" %
             (accuracy, recall, precision, f))
        
    return 




#1: Logistic Regression
run_model(LogisticRegression(),'Logit',[{'penalty': ['l1','l2'],'C': [0.01,0.5,1,5]}])




#2: Random Forest
run_model(RandomForestClassifier(n_jobs=-1),'RF',[{'n_estimators':[10,50,100,150,200,250]}])




#3: Extra Trees Classifier
run_model(ExtraTreesClassifier(n_jobs=-1),'ET',[{'n_estimators':[10,50,100,150,200,250,300,500]}])




#4: SVM Classifier
run_model(SVC(C=1),'SVM',[{'kernel': ['linear','rbf','poly'],'C': [0.1,0.5,1, 10, 50]}])

