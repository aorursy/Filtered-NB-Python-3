#!/usr/bin/env python
# coding: utf-8



# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:07:41 2017


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import scale
from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm

df= pd.read_csv('../input/boston.csv')
print(df.head())

corr=df.corr()
sns.heatmap(corr,annot=True,linewidths=1)
plt.show()

X = df.iloc[:,0:13]
Y = df.iloc[:,-1]
names = df.columns.values
print(names)
print(X)
print(Y)
dt=DecisionTreeRegressor()
print(dt.fit(X,Y))

# Print the results
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), dt.feature_importances_), names), reverse=True))
# Isolate feature importances 
importance = dt.feature_importances_

sorted_importances = np.argsort(importance)

# Insert padding
padding = np.arange(len(names)-1) + 0.5

# Plot the data
plt.barh(padding, importance[sorted_importances], align='center')

# Customize the plot
plt.yticks(padding, names[sorted_importances])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")

# Show the plot
plt.show()

# Import `PCA` from `sklearn.decomposition`
from sklearn.decomposition import PCA

# Build the model
pca = PCA(n_components=4)

# Reduce the data, output is ndarray
reduced_data = pca.fit_transform(X,Y)

# Inspect shape of the `reduced_data`
reduced_data.shape

# print out the reduced data
print(reduced_data)


#  k-NN 
n_neig = 5 
# Set sc = True if you want to scale your features 
sc = False 


X = df.drop('MV' , 1).values 
 
# Here we scale, if desired 
if sc == True: X = scale(X) 
# Target value
y1 = df['MV'].values 

y = y1 <= 5 
# new target variable: is the rating <= 5? 
# Split the data into a test set and a training set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Train k-NN model and print performance on the test set 
knn = KNeighborsClassifier(n_neighbors = n_neig) 
knn_model = knn.fit(X_train, y_train) 
y_true, y_pred = y_test, knn_model.predict(X_test)
print('k-NN accuracy for test set: %f' % knn_model.score(X_test, y_test)) 
print(classification_report(y_true, y_pred)) 




class LinearRegression(linear_model.LinearRegression):
    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())







