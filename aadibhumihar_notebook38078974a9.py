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




from sklearn.linear_model import LogisticRegression
train_df = pd.read_csv('train.csv')
train_m = train_df.as_matrix()




import pandas as pd
from sklearn.linear_model import LogisticRegression
train_df = pd.read_csv('train.csv')
train_m = train_df.as_matrix()
X = train_m[:50,1:]
y = train_m[:50,0]




import pandas as pd
from sklearn.linear_model import LogisticRegression
train_df = pd.read_csv('train.csv')




print('Aditya')




import os, sys
path = os.getcwd();




import os, sys
path = os.getcwd();
print("Folder item :",os.listdir(path))




import pandas as pd
from sklearn.linear_model import LogisticRegression
train_df = pd.read_csv('../input/train.csv')




train_m = train_df.as_matrix()
X = train_m[:50,1:]
y = train_m[:50,0]
logistic = LogisticRegression()
logistic.fit(X,y)




train_m = train_df.as_matrix()
X = train_m[:50,1:]
y = train_m[:50,0]
logistic = LogisticRegression()
logistic.fit(X,y)




train_m = train_df.as_matrix()
X = train_m[:500,1:]
y = train_m[:500,0]
logistic = LogisticRegression()
logistic.fit(X,y)
#print ('Predicted class %s, real class %s' % (logistic.predict(X[419,:]),y[419]))




train_m = train_df.as_matrix()
X = train_m[:500,1:]
y = train_m[:500,0]
logistic = LogisticRegression()
logistic.fit(X,y)
#print ('Predicted class %s, real class %s' % (logistic.predict(X[419,:]),y[419]))






