#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import seaborn as sns
import random as rd
import scipy

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.grid_search import GridSearchCV

import xgboost as xgb

#Matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (3, 3)




df = pd.read_csv('../input/creditcard.csv')
print('The dataset contains %d entries' % (len(df)))




#The dataset is really unbalanced
nb_fraud = df['Class'].value_counts()[1]
print('Fraud represents only %0.3f%% of the whole dataset' % (float(nb_fraud)/len(df)*100))




numerical_features = ['Time', 'Amount'] + ['V%d'%i for i in range(1,29)]
target = ['Class']
all_variables = numerical_features + target




y = df[target].values
X = df.drop('Class',axis=1)




X.describe()




X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_valid.shape)




# rescaling 
scaler= StandardScaler() 
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features].values)
X_valid[numerical_features] = scaler.transform(X_valid[numerical_features].values)




# declare classifier 
clf1 = LogisticRegression() # pimp me 
clf2 = RandomForestClassifier(n_estimators =100, max_depth = 10, class_weight = 'auto') # pimp me 
clf3 = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)




# train model 1
clf1.fit(X_train,y_train)




# train model 2
clf2.fit(X_train,y_train.ravel())




clf3.fit(X_train, y_train.ravel())




probas = {}




# evaluate and plot roc curve 1
probas['lr'] = clf1.predict_proba(X_valid)
fpr, tpr, thresholds = roc_curve(y_valid, probas['lr'][:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % roc_auc)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




# evaluate and plot roc curve 2
probas['rf'] = clf2.predict_proba(X_valid)
fpr, tpr, thresholds = roc_curve(y_valid, probas['rf'][:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % roc_auc)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




# evaluate and plot roc curve 2
probas['xgb'] = clf3.predict_proba(X_valid)
fpr, tpr, thresholds = roc_curve(y_valid, probas['xgb'][:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % roc_auc)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()




# Scoring Metrics
y_pred = {
    'lr' : clf1.predict(X_valid),
    'rf' : clf2.predict(X_valid),
    'xgb' : clf3.predict(X_valid)
}

# Scores
precision = {}
recall = {}
f1 = {}
for clf in ['lr', 'rf', 'xgb']:
    precision[clf] = precision_score(y_valid, y_pred[clf])
    recall[clf] = recall_score(y_valid, y_pred[clf])
    f1[clf] = f1_score(y_valid, y_pred[clf])
    print ('For classifier %s :\n\t precision = %0.3f\n\t recall = %0.3f\n\t f1 =  %0.3f' % (clf, precision[clf], recall[clf], f1[clf]))




# Set the parameters by cross-validation
tuned_parameters = [ {'reg_alpha' : [1],
                      'gamma' : [0.1, 0.5]}]

scores = ['f1']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(xgb.XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train.ravel())

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_valid, clf.predict(X_valid)
    print(classification_report(y_true, y_pred))
    print()




clf.__dict__




y_pred2 = clf.predict(X_valid)
precision['grid'] = precision_score(y_valid, y_pred2)
recall['grid'] = recall_score(y_valid, y_pred2)
f1['grid'] = f1_score(y_valid, y_pred2)




c = 'grid'
print ('For classifier %s :\n\t precision = %0.3f\n\t recall = %0.3f\n\t f1 =  %0.3f' % (c, precision[c], recall[c], f1[c]))






