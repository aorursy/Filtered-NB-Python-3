#!/usr/bin/env python
# coding: utf-8



import numpy as np
import sklearn
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt




data_df = pd.read_csv('../input/creditcard.csv')




data_df.head(2)




#check if the df contains any NaN values

data_df.isnull().values.any()




classes = data_df.Class.unique()
print(classes)




print('class corresponding to non-fraud', classes[0],': ', len(data_df[data_df.Class==classes[0]]))
print('class corresponding to fraud', classes[1],': ', len(data_df[data_df.Class==classes[1]]))




#transform the dataframe to an array
data = data_df.as_matrix()

X_data = data[:,:(data_df.shape[1]-1)]
y_data = data[:,(data_df.shape[1]-1)]




#split the data into training and test data
from sklearn.model_selection import train_test_split

X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(X_data, y_data, test_size=0.25)




#scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_data_train = scaler.fit_transform(X_data_train)
X_data_test = scaler.fit_transform(X_data_test)




#train using neural networks
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


#fit decision tree classifier
model_nn = MLPClassifier(hidden_layer_sizes=2, max_iter=2000)
model_nn.fit(X_data_train, y_data_train)

#predict 'y' for test data
y_data_pred_test = model_nn.predict(X_data_test)

#score
print('Accuracy: ', accuracy_score(y_data_test, y_data_pred_test))
print('confusion matrix:', confusion_matrix(y_data_test, y_data_pred_test))
print('F1:', f1_score(y_data_test, y_data_pred_test))




#this is the data corresponding to fraud
data_fraud_df = data_df[data_df.Class==classes[1]]
data_fraud_df = data_fraud_df.reset_index(drop=True)
data_fraud = data_fraud_df.as_matrix()


#this is the data corresponding to non-fraud
data_nonfraud_df = data_df[data_df.Class==classes[0]]
data_nonfraud_df = data_nonfraud_df.reset_index(drop=True)
data_red_nonfraud_df =     data_nonfraud_df.ix[np.random.random_integers(1, max(data_nonfraud_df.index),max(data_fraud_df.index)+1)]
data_red_nonfraud_df = data_red_nonfraud_df.reset_index(drop=True)




#now lets join both fraud and non-fraud of the same length
data_red_df = pd.concat([data_red_nonfraud_df, data_fraud_df])
data_red = data_red_df.as_matrix()

X_data_red = data_red[:,:(data_red_df.shape[1]-1)]
y_data_red = data_red[:,(data_red_df.shape[1]-1)]




#define train and test of the symmetric data
X_data_red_train, X_data_red_test, y_data_red_train, y_data_red_test =            train_test_split(X_data_red, y_data_red, test_size=0.25)




#scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_data_red_train = scaler.fit_transform(X_data_red_train)
X_data_red_test = scaler.fit_transform(X_data_red_test)




#again, lets use neural networks

model_nn.fit(X_data_red_train, y_data_red_train)

#predict 'y' for test data
y_data_red_pred_test = model_nn.predict(X_data_red_test)

#score
print('Accuracy: ', accuracy_score(y_data_red_test, y_data_red_pred_test))
print('confusion matrix:', confusion_matrix(y_data_red_test, y_data_red_pred_test))
print('F1:', f1_score(y_data_red_test, y_data_red_pred_test))




#lets look at the cross validation
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem

cv = KFold(len(y_data_red_pred_test), 5, shuffle=True, random_state=0)
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(model_nn, X_data_red_test, y_data_red_pred_test, cv=cv)
print(scores)
print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))




from sklearn.manifold import Isomap
iso = Isomap(n_neighbors=30, n_components=2)

#project the data to 2-dimension features
iso.fit(X_data_red_train[:50,:])
Xdata_red_projected = iso.transform(X_data_red_train)

#visualise the data
plt.scatter(Xdata_red_projected[:, 0], Xdata_red_projected[:, 1], c=y_data_red_train,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral'))

plt.clim(-0.5, 9.5);




get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
sns.pairplot(data_red_df, hue='Class',vars=['Time', 'Amount']);




sns.pairplot(data_red_df, hue='Class',vars=['V1','V2','V3','V4', 'V5', 'V6','V7','V8','V9','V10']);




sns.pairplot(data_red_df, hue='Class',vars=[ 'V11', 'V12','V13','V14','V15','V16','V17','V18','V19','V20','V21']);




sns.pairplot(data_red_df, hue='Class',vars=[ 'V22','V23','V24','V25','V26','V27','V28']);




sns.pairplot(data_red_df, hue='Class',vars=['V10','V14','V16','V17']);




data_card_df = pd.concat([data_red_df['V10'],data_red_df['V16'],data_red_df['V14'],                            data_red_df['V17'],data_red_df['Class']],axis=1)


data_card = data_card_df.as_matrix()

X = data_card[:,:(data_card_df.shape[1]-1)]
y = data_card[:,(data_card_df.shape[1]-1)]




iso.fit(X[:50,:])
data_projected = iso.transform(X)
data_projected.shape




plt.scatter(data_projected[:, 0], data_projected[:, 1], c=y,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral'))

plt.clim(-0.5, 9.5);




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)




from sklearn.tree import DecisionTreeClassifier

#fit decision tree classifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)

#predict 'y' for test data
y_pred_test_dt = model_dt.predict(X_test)

#score
print(confusion_matrix(y_test, y_pred_test_dt))
print(f1_score(y_test, y_pred_test_dt))




#fit data
model_nn.fit(X_train, y_train)

#predict y
y_pred_test_nn = model_nn.predict(X_test)

#score
print(confusion_matrix(y_test, y_pred_test_nn))
print(f1_score(y_test, y_pred_test_nn))




from sklearn.ensemble import RandomForestClassifier

#fit
model_rf = RandomForestClassifier(criterion='entropy')
model_rf.fit(X_train, y_train)

#predict y
y_pred_test_rf = model_rf.predict(X_test)

#score
print(confusion_matrix(y_test, y_pred_test_rf))
print(f1_score(y_test, y_pred_test_rf))




cv = KFold(len(y), 5, shuffle=True, random_state=0)

scores_dt = cross_val_score(model_dt, X, y, cv=cv)
print(scores_dt)
print("Mean score decision tree: {0:.3f} (+/-{1:.3f})".format(np.mean(scores_dt), sem(scores_dt)))


scores_nn = cross_val_score(model_nn, X, y, cv=cv)
print(scores_nn)
print("Mean score neural networks: {0:.3f} (+/-{1:.3f})".format(np.mean(scores_nn), sem(scores_nn)))
scores_rf = cross_val_score(model_rf, X, y, cv=cv)
print(scores_rf)
print("Mean score random forest: {0:.3f} (+/-{1:.3f})".format(np.mean(scores_rf), sem(scores_rf)))






