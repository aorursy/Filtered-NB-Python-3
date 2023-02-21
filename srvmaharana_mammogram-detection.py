#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




import pandas as pd

masses_data = pd.read_csv('../input/mammographic_masses.data.txt',names=['BI_RADS', 'age', 'shape', 'margin', 'density','severity'])
masses_data.head(5)




masses_data.replace(to_replace='?', value='NaN', inplace=True)




masses_data.head()




masses_data = pd.read_csv('../input/mammographic_masses.data.txt', na_values=['?'], names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
masses_data.head()




masses_data.describe()




masses_data.loc[(masses_data['age'].isnull()) |
              (masses_data['shape'].isnull()) |
              (masses_data['margin'].isnull()) |
              (masses_data['density'].isnull())]




masses_data.dropna(inplace=True)
masses_data.describe()




masses_data.drop('BI-RADS',inplace=True,axis=1)
masses_data.head()




all_features = masses_data.drop('severity',axis=1)


all_classes = masses_data['severity']

feature_names = ['age', 'shape', 'margin', 'density']

all_features.head()




all_classes.head()




sns.distplot(all_features['age'],bins=40)




from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)
all_features_scaled




all_features_scaled=pd.DataFrame(all_features_scaled,columns=feature_names)
all_features_scaled.head()




import numpy
from sklearn.model_selection import train_test_split

numpy.random.seed(1234)

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(all_features_scaled, all_classes, train_size=0.75, random_state=1)




from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier(random_state=1)

# Train the classifier on the training set
clf.fit(training_inputs, training_classes)




#from IPython.display import Image  
#from sklearn.externals.six import StringIO  
#from sklearn import tree
#from pydotplus import graph_from_dot_data 

#dot_data = StringIO()  
#tree.export_graphviz(clf, out_file=dot_data,feature_names=feature_names)  
#graph = graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())  




clf.score(testing_inputs, testing_classes)




from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

cv_scores.mean()




from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=1)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

cv_scores.mean()




from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear', C=C)




cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)

cv_scores.mean()




from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=10)
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)

cv_scores.mean()




for n in range(1, 50):
    clf = neighbors.KNeighborsClassifier(n_neighbors=n)
    cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
    print (n, cv_scores.mean())




from sklearn.naive_bayes import MultinomialNB

scaler = preprocessing.MinMaxScaler()
all_features_minmax = scaler.fit_transform(all_features)

clf = MultinomialNB()
cv_scores = cross_val_score(clf, all_features_minmax, all_classes, cv=10)

cv_scores.mean()




C = 1.0
svc = svm.SVC(kernel='rbf', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()




C = 1.0
svc = svm.SVC(kernel='sigmoid', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()




C = 1.0
svc = svm.SVC(kernel='poly', C=C)
cv_scores = cross_val_score(svc, all_features_scaled, all_classes, cv=10)
cv_scores.mean()




from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
cv_scores = cross_val_score(clf, all_features_scaled, all_classes, cv=10)
cv_scores.mean()




from keras.layers import Dense
from keras.models import Sequential

def create_model():
    model = Sequential()
    #4 feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    # "Deep learning" turns out to be unnecessary - this additional hidden layer doesn't help either.
    #model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # Output layer with a binary classification (benign or malignant)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model; rmsprop seemed to work best
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model




all_features_scaled.values




from keras.wrappers.scikit_learn import KerasClassifier

# Wrap our Keras model in an estimator compatible with scikit_learn
estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, verbose=0)
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, all_features_scaled.values, all_classes, cv=10)
cv_scores.mean()




model = Sequential()
#4 feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)
model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
# "Deep learning" turns out to be unnecessary - this additional hidden layer doesn't help either.
#model.add(Dense(4, kernel_initializer='normal', activation='relu'))
# Output layer with a binary classification (benign or malignant)
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model; rmsprop seemed to work best
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])




history = model.fit(training_inputs.values,training_classes,batch_size=100,epochs=16,verbose=2,validation_data=(testing_inputs.values,testing_classes))






