#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Seaborn for plotting
# and ignore all warnings
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
plt.style.use('ggplot')

data = pd.read_csv('../input/diabetes.csv')
data.shape
data.head()




grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'Glucose', 'Insulin')
grid.add_legend()
plt.show()

# High risk: insulin_geq_400




grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'Glucose', 'BMI')
grid.add_legend()
plt.show()

# High risk: BMI_geq_48




grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'Glucose', 'DiabetesPedigreeFunction')
grid.add_legend()
plt.show()

# High risk: pedigree_geq_1

# From all scatterplots
# High risk: glucose_geq_170




grid = sns.FacetGrid(data, hue='Outcome')
grid.map(plt.scatter, 'BloodPressure', 'SkinThickness')
grid.add_legend()
plt.show()

# High risk: blood_geq_92




n_bins = data.Pregnancies.max() - data.Pregnancies.min()
grid = sns.FacetGrid(data, row='Outcome')
grid.map(plt.hist, 'Pregnancies', bins=np.arange(0, n_bins)-0.5)
plt.show()

# Low risk: preg_1_or_2




n_bins = data.Age.max() - data.Age.min()
grid = sns.FacetGrid(data, row='Outcome')
grid.map(plt.hist, 'Age', bins=np.arange(20, n_bins)-0.5)
plt.show()

# Low risk: age_leq_28
# High risk: age_52_or_53




data = data[data.SkinThickness != 0]
n_bins = data.SkinThickness.max() - data.SkinThickness.min()
grid = sns.FacetGrid(data, row='Outcome')
grid.map(plt.hist, 'SkinThickness', bins=np.arange(n_bins)-0.5)
plt.show()

# Low risk: st_10_to_23




from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score

X = data.ix[:, :-1]
y = data.ix[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Instantiate logistic regression
clf = LogisticRegression()

#Cross-validate logistic regression
print("\nLogistic regression results for 10-fold cross-validation:\n")
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
print(("Accuracies:\n %s\n\n" +
       "Best accuracy on held-out data: %.4f\n\n" +
       "Mean accuracy on held-out data: %.4f\n\n") % (str(scores), scores.max(), scores.mean()))
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc')
print(("AUC:\n %s\n\n" +
       "Best AUC on held-out data: %.4f\n\n" + 
       "Mean AUC on held-out data: %.4f\n\n") % (str(scores), scores.max(), scores.mean()))

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("="*80)
print("\nLogistic regression performance on unseen data:")
print("\nlog-loss: %.4f" % log_loss(y_test, pred))
print("\nAUC: %.4f" % roc_auc_score(y_test, pred))
print("\nF1 score: %.4f" % f1_score(y_test, pred))
print("\nAccuracy: %.4f" % accuracy_score(y_test, pred))
print("="*80)




######################################## 
# Create binary features
######################################## 

# Insulin >= 400
X['insulin_geq_400'] = np.where(X['Insulin'] >= 400, 1, 0)

# BMI >= 48
X['bmi_geq_48'] = np.where(X['BMI'] >= 48, 1, 0)

# Diabetes Pedigree Function >= 1
X['pedigree_geq_1'] = np.where(X['DiabetesPedigreeFunction'] >= 1.0, 1, 0)

# Glucose >= 170
X['glucose_geq_170'] = np.where(X['Glucose'] >= 170, 1, 0)

# Blood Pressure >= 92
X['blood_geq_92'] = np.where(X['BloodPressure'] >= 92, 1, 0)

# One or two pregnancies
X['preg_1_or_2'] = np.where(X['Pregnancies'] == 1, 1, 0) +                      np.where(X['Pregnancies'] == 2, 1, 0)

# Age <= 28
X['age_leq_28'] = np.where(X['Age'] <= 28, 1, 0)

# Age is 52 or 53
X['age_52_or_53'] = np.where(X['Age'] == 52, 1, 0) +                       np.where(X['Age'] == 53, 1, 0)

# 10 <= Skin Thickness >= 23
X['skin_10_to_23'] = np.where(X['SkinThickness'] >= 10, 1, 0) -                        np.where(X['SkinThickness'] > 23, 1, 0)

X.head()




from sklearn.model_selection import train_test_split

# Isolate a training set for CV.
# Testing set won't be touched until CV is complete.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Note that the train/test split *must* be done before any imputation.
# If we impute values before isolating the training set,
# then values from the testing set might 
# influence the imputed values in the training set.
# See Abu-Mostafa's "Learning from Data" to read more about this issue,
# sometimes referred to as "data snooping".




normals = [0]*3
variables = ['Glucose', 'SkinThickness', 'BMI']

# Generate imputation values with Gaussian randomness.
for n, v in zip(range(len(normals)), variables):
    # Shift the mean up to account for skewness caused by zeros.
    v_mean = X_train[v].mean()*1.5

    # Use surrogate deviation.
    # (Sometimes I get strange values when using .std(). Why?)
    v_std = v_mean*0.1

    normals[n] = np.random.normal(loc = v_mean, scale = v_std)

print("Imputing zeros in Glucose, SkinThickness, and BMI with")
print("%f, %f, and %f" % (normals[0], normals[1], normals[2]))

# Impute.
X_train = X_train.replace(to_replace = {'Glucose': {0: normals[0]}, 
                                  'SkinThickness': {0: normals[1]}, 
                                  'BMI': {0: normals[2]}})




normals = [0]*3
variables = ['Glucose', 'SkinThickness', 'BMI']

# Generate imputation values with Gaussian randomness.
for n, v in zip(range(len(normals)), variables):
    # Shift the mean up to account for skewness caused by zeros.
    v_mean = X_test[v].mean()*1.5

    # Use surrogate deviation.
    # (Sometimes I get strange values when using .std(). Why?)
    v_std = v_mean*0.1

    normals[n] = np.random.normal(loc = v_mean, scale = v_std)

print("Imputing zeros in Glucose, SkinThickness, and BMI with")
print("%f, %f, and %f" % (normals[0], normals[1], normals[2]))

# Impute.
X_test = X_test.replace(to_replace = {'Glucose': {0: normals[0]}, 
                                  'SkinThickness': {0: normals[1]}, 
                                  'BMI': {0: normals[2]}})




from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, f1_score, accuracy_score

# Instantiate logistic regression
clf = LogisticRegression()

#Cross-validate logistic regression
print("\nLogistic regression results for 10-fold cross-validation:\n")
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
print(("Accuracies:\n %s\n\n" +
       "Best accuracy on held-out data: %.4f\n\n" +
       "Mean accuracy on held-out data: %.4f\n\n") % (str(scores), scores.max(), scores.mean()))
scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc')
print(("AUC:\n %s\n\n" +
       "Best AUC on held-out data: %.4f\n\n" + 
       "Mean AUC on held-out data: %.4f\n\n") % (str(scores), scores.max(), scores.mean()))




clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("="*80)
print("\nLogistic regression performance on unseen data:")
print("\nlog-loss: %.4f" % log_loss(y_test, pred))
print("\nAUC: %.4f" % roc_auc_score(y_test, pred))
print("\nF1 score: %.4f" % f1_score(y_test, pred))
print("\nAccuracy: %.4f" % accuracy_score(y_test, pred))
print("="*80)

