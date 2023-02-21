#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# tweaks for the imported libraries...
# tweaks for Numpy & Pandas
#pd.set_option('display.notebook_repr_html',True)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)
pd.set_option('display.width',1024)
# force all numoy & pandas floating point output to 3 decimal places
float_formatter = lambda x: '%.3f' % x
np.set_printoptions(formatter={'float_kind':float_formatter})
pd.set_option('display.float_format', float_formatter)
# force Numpy to display very small floats using floating point notation
np.set_printoptions(threshold=np.inf)
# force GLOBAL floating point output to 3 decimal places
get_ipython().run_line_magic('precision', '3')

# tweaks for plotting libraries (Matplotlib & Seaborn) [recommended]
plt.style.use('seaborn-muted')
sns.set_context(context='notebook',font_scale=1.0)
sns.set_style('whitegrid')

seed = 42 #sum(map(ord, 'Kaggle - Pima Indians Diabetes Analysis'))
np.random.seed(seed)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




data = pd.read_csv("../input/data.csv",index_col=0)
data.head()




# lets view some statistics on the dataset
data.describe()




data.info()




# do we have any NUlls?
data.isnull().sum()




# plot the distribution of the target column (diagnosis)
f, ax = plt.subplots(figsize=(6,4))
_ = sns.countplot(data=data,x='diagnosis',ax=ax)




# save names of features & outcome in variables we can use later
features = data.columns.values[1:-1]
outcome = data.columns.values[0]
features, outcome




# plot distribution of all features
with sns.axes_style('ticks'):
    _ = data[features].hist(edgecolor='k',grid=False, figsize=(20,15))




# split the data into train/test sets
from sklearn.model_selection import train_test_split

# The outcome variable distribution is imbalanced (B records are ~1.6 times M records)
# let's stratify the selection, so we have an equal chance of getting 0 & 1 outcomes in train/test sets.
train_set, test_set =   train_test_split(data, test_size=0.20, random_state=0,stratify=data[outcome])
X_train = train_set[features]
X_test = test_set[features]
y_train = train_set[outcome]
y_test = test_set[outcome]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)




y_train[:3]




# Label encode the output (diagnosis) column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = pd.Series(y_train, index=X_train.index)

y_test = le.fit_transform(y_test)
y_test = pd.Series(y_test, index=X_test.index)




y_train[:3]




from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, f1_score




# utility functions
def do_kfold_cv(classifier, X_train, y_train, n_splits=10, scoring='roc_auc'):
    """ do a k-fold cross validation run on classifier & training data
      and return cross-val scores """   
    kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(classifier, X_train, y_train, scoring=scoring, cv=kfolds)
    return cv_scores

def test_classifier(clf_tuple, X_train, y_train, X_test, y_test, scoring='roc_auc', verbose=2):
    """ run a k-fold test, fit model to training data & calculates some classification metrics
        like accuracy_score, f1-score and AUC score """
    # extract classifier instance & name
    classifier, classifier_name = clf_tuple
   
    if verbose > 0:
        print('Testing classifier %s...' % classifier_name)
    
    classifier.fit(X_train, y_train)
    
    # accuracy scores, against test data
    acc_score = classifier.score(X_test, y_test)
    
    # k-fold cross-validation scores
    cv_scores = do_kfold_cv(classifier, X_train, y_train, scoring=scoring)

    # roc-auc score
    y_pred_proba_train = classifier.predict_proba(X_test)[:,1]
    auc_score = roc_auc_score(y_test, y_pred_proba_train)
    
    # F1 score
    f1score = f1_score(y_test,classifier.predict(X_test))

    if verbose > 1:   
        print('   - cross-val score : Mean - %.3f Std - %.3f Min - %.3f Max - %.3f' %                   (np.mean(cv_scores), np.std(cv_scores), np.min(cv_scores), np.max(cv_scores)))
        print('   - accuracy score  : %.3f' % (acc_score))
        print('   - AUC score       : %.3f' % (auc_score))
        print('   - F1 score        : %.3f' % (f1score))
              
    return cv_scores, acc_score, auc_score, f1score

def test_classifiers(clf_list, X_train, y_train, X_test, y_test, scoring='roc_auc', verbose=2):
    """ run a list of classifiers against the training & test sets and
        return a pandas DataFrame of scores """
    classifier_names = []
    clf_cv_scores = []
    clf_acc_scores = []
    clf_auc_scores = []
    clf_f1_scores = []
        
    for clf_tuple in clf_list:
        cv_scores, acc_score, auc_score, f1score =             test_classifier(clf_tuple, X_train, y_train, X_test, y_test, scoring=scoring, verbose=verbose)
        classifier, classifier_name = clf_tuple
        classifier_names.append(classifier_name)
        clf_cv_scores.append(np.mean(cv_scores))
        clf_acc_scores.append(acc_score)
        clf_auc_scores.append(auc_score)
        clf_f1_scores.append(f1score)
   
    # now create a DataFrame of all the scores & return
    scores_df = pd.DataFrame(data=clf_cv_scores, index=classifier_names,
                             columns=['mean_cv_scores'])
    scores_df['accuracy_scores_test'] = clf_acc_scores
    scores_df['auc_scores_test'] = clf_auc_scores
    scores_df['f1_scores'] = clf_f1_scores
    return scores_df




from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier




# instantiate the classifiers 
# We will create several classifiers, which will be run on the training/test sets to analyze performance
# Since we have 569 total observations (samples) in the dataset and 30 features, we will use PCA to select
# N features (defaulting to 10). All this can be done in one 'Pipeline' object
def setup_classifiers(n_comps=10):
    # NOTE: n_comps is the number of components PCA should pick
    clf_list = []
    N_COMPS = n_comps
    
    if ((N_COMPS < 1) | (N_COMPS > len(features))):
        raise ValueError('n_comps is out of range! Expecting value between 1 and %d' % len(features))

    # KNN classifier
    pipe_knn = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=N_COMPS)),
                         ('clf', KNeighborsClassifier(n_neighbors=5))])
    clf_list.append((pipe_knn, 'KNN Classifier'))

    # Logistic Regression classifier
    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=N_COMPS)),                    
                        ('clf', LogisticRegression(penalty='l2', C=1.0, random_state=seed))])
    clf_list.append((pipe_lr, 'LogisticRegression Classifier'))

    # SVC (Linear) classifier
    pipe_svcl = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=N_COMPS)),                    
                         ('clf', SVC(kernel='linear',C=1.0, gamma='auto', probability=True, random_state=seed))])
    clf_list.append((pipe_svcl, 'SVC(Linear) Classifier'))

    # SVC (Gaussian) classifier
    pipe_svcg = Pipeline([('scl', StandardScaler()),
                          ('pca', PCA(n_components=N_COMPS)),                         
                          ('clf', SVC(kernel='rbf',C=1.0, gamma='auto', probability=True, random_state=seed))])
    clf_list.append((pipe_svcg, 'SVC(Gaussian) Classifier')) 

    # Naive Bayes - don't need scaling
    pipe_nb = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=N_COMPS)),                    
                        ('clf', GaussianNB())])
    clf_list.append((pipe_nb, 'Naive Bayes Classifier')) 

    # DecisionTree classifier
    pipe_dt = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=N_COMPS)),                    
                        ('clf', DecisionTreeClassifier(random_state=seed, max_depth=5))])
    clf_list.append((pipe_dt, 'Decision Tree Classifier'))

    # ExtraTrees classifier
    pipe_xtc = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=N_COMPS)),                    
                         ('clf', ExtraTreesClassifier(max_depth=5, n_estimators=100, random_state=seed))])
    clf_list.append((pipe_xtc, 'Extra Trees Classifier'))

    # Random Forest classifier
    pipe_rfc = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=N_COMPS)),                    
                         ('clf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed))])
    clf_list.append((pipe_rfc, 'Random Forests Classifier'))

    # Gradient boosting classifier    
    pipe_gbc = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=N_COMPS)),                    
                         ('clf', GradientBoostingClassifier(loss='deviance',learning_rate=0.1,
                                                            n_estimators=100, max_depth=5, random_state=seed))])
    clf_list.append((pipe_gbc, 'Gradient Boosting Classifier'))
    
    return clf_list




# run the classifiers on the train/test data & report metrics
clf_list = setup_classifiers()
scores_df = test_classifiers(clf_list, X_train, y_train, X_test, y_test, verbose=2)
print('Done!')




scores_df.sort_values(by=['accuracy_scores_test'], ascending=False, inplace=True)
print('\nClassifiers sorted by Accuracy Scores on test data (descending):')
scores_df




#print(confusion_matrix(y_test, pipe_lr.predict(X_test)))

# following is a better way of viewing the confusion matrix
# we will first inverse transform the y_test & y_pred vectors so we can see 'M' and 'B' in 
# the confusion matrix
y_test2 = le.inverse_transform(y_test)
pipe_lr = clf_list[1][0]
y_pred2 = le.inverse_transform(pipe_lr.predict(X_test))
# this shows the same result as confusion_matrix() call, but displayed better
pd.crosstab(y_test2.ravel(), y_pred2, rownames=['Actual'], colnames=['Predicted->'], margins=False)




num_features = len(features)
best_n, best_score, best_classifier = 0, 0.0, ''

for n in range(5,16):
    clf_list = setup_classifiers(n_comps=n)
    scores_df = test_classifiers(clf_list, X_train, y_train, X_test, y_test, verbose=0)
    scores_df.sort_values(by=['accuracy_scores_test'], ascending=False, inplace=True)
    print('\nClassifiers sorted by Accuracy Scores (descending) - for %d components:' % n)
    print(scores_df)
    top_score = scores_df.iloc[0]['accuracy_scores_test']
    if top_score > best_score:
        best_score = top_score
        best_n = n
        best_classifier = scores_df.iloc[0].name
        
print('\nBest results: with %s - PCA = %d, Accuracy score = %.4f' % (best_classifier, best_n, best_score))




# now let us create the classifier basedon above results & re-check
pipe_lr_best = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=best_n)),                    
                         ('clf', LogisticRegression(penalty='l2', C=1.0, random_state=seed))])
scores = test_classifier((pipe_lr_best, 'Logistic Regression Classifier - %d PCA' % best_n), 
                         X_train, y_train, X_test, y_test, verbose=2)
# Expect it to display same results as last output row in output above...




# and confusion matrix for the classifier with 14 PCA
y_test3 = le.inverse_transform(y_test)
y_pred3 = le.inverse_transform(pipe_lr_best.predict(X_test))
# this shows the same result as confusion_matrix() call, but displayed better
pd.crosstab(y_test3.ravel(), y_pred3, rownames=['Actual'], colnames=['Predicted->'], margins=False)




# let's view a plot of explained variance ratios
with sns.axes_style('ticks'):
    x_range = range(1,best_n+1)
    plt.bar(x_range, pipe_lr_best.named_steps['pca'].explained_variance_ratio_, 
            alpha=0.8, align='center', label='Explained Variance')
    plt.legend(loc='best')
    plt.xticks(x_range)
    plt.show()
    plt.close()






