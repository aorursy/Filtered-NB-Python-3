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




test  = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')




#Data Exploration




print(train.shape)
train.head()




print(test.shape)
test.head()




import matplotlib.pyplot as plt
plt.hist(train["label"])
plt.title("Frequency Histogram of Numbers in Training Data")
plt.xlabel("Number Value")
plt.ylabel("Frequency")
plt.show()




label_train=train['label']
train=train.drop('label', axis=1)




train.head()




# data normalisation
train = train/255
test = test/255




from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, label_train, train_size = 0.8,random_state = 42)




#PCA
from sklearn import decomposition
pca = decomposition.PCA(n_components=50)
pca.fit(X_train)
PCtrain = pca.transform(X_train)
PCval = pca.transform(X_val)
PCtest = pca.transform(test)




X_train= PCtrain




X_cv = PCval




#train
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=1e-5,max_iter=500,learning_rate='invscaling',
                    hidden_layer_sizes=(150,), random_state=1)

clf.fit(X_train,y_train)




#predict
predicted = clf.predict(X_cv)
expected = y_val




print(predicted[0:30])




from sklearn import  metrics
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))




print('accurcy :',metrics.accuracy_score(expected, predicted))




output_label = clf.predict(PCtest)




output = pd.DataFrame(output_label,columns = ['Label'])
output.reset_index(inplace=True)
output['index'] = output['index'] + 1
output.rename(columns={'index': 'ImageId'}, inplace=True)
output.to_csv('output.csv', index=False)
output.head()

