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

from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = pd.read_csv('kaggle/stocknews/Combined_News_DJIA.csv')
spy = pd.read_csv('kaggle/stocknews/SPY spot.csv')

spy_spot = [round(float(spy['SPY.Adjusted'][spy['index.spot.'] == i]),4) for i in data['Date']]
data['SPY'] = spy_spot

#train = data[data['Date'] < '20140101']
#test = data[data['Date'] > '20131231']

trainheadlines = []
for row in range(0,len(data.index)):
    trainheadlines.append(' '.join(str(x) for x in data.iloc[row,2:27]))
    


'-----------------------'
reset_model = 100
train_length =  1000
threshold = 0.6
'-----------------------'



d1 = data.shape[0]

spy_sign = [int(np.sign(data['SPY'][i+1] - data['SPY'][i])/2+0.5) for i in range(0,d1-1)] + [1]*1
data['SPY Sign'] = spy_sign
daily_predict = [0.5]*d1
daily_position = [0] * d1

for i in range(train_length,d1,reset_model):
#for i in range(train_length+reset_model,550,reset_model):
    advancedvectorizer = TfidfVectorizer(ngram_range=(2,2))
    #advancedmodel = LogisticRegression()
    #get training data
    temp_trainheadlines = trainheadlines[i-train_length:i]
    temp_response = data['SPY Sign'][i-train_length:i]
    temp_train = advancedvectorizer.fit_transform(temp_trainheadlines)
    
    #train on SVM
    clf = SVC(probability=True, kernel='rbf')
    clf.fit(temp_train, temp_response)
    #advancedmodel = advancedmodel.fit(temp_advancedtrain, temp_response)
    
    #get testing data
    today_headlines = trainheadlines[i:min(i+reset_model,d1)]
    today_advancedpredict = advancedvectorizer.transform(today_headlines)
    
    predictions_prob = clf.predict_proba(today_advancedpredict)
    
    #take position based on probability
    long_temp = np.array([1 if j[1]>threshold else 0 for j in predictions_prob])
    short_temp = np.array([-1 if j[0]>threshold else 0 for j in predictions_prob])
    daily_position[i:min(i+reset_model,d1)] = long_temp+short_temp
    
    #today_prediction = advancedmodel.predict(today_advancedpredict)
    #advancedmodel.predict(temp_advancedtrain)
    long_temp = np.array([1 if j[1]>0.5 else 0 for j in predictions_prob])
    short_temp = np.array([0 if j[0]>0.5 else 0 for j in predictions_prob])
    daily_predict[i:min(i+reset_model,d1)] = long_temp+short_temp
    print str(i) + " day completed"


spy_return = np.array([data['SPY'][i+1]/data['SPY'][i] for i in range(0,d1-1)] + [1]) - 1.0
pnl = spy_return * np.array(daily_position)
#plot pnl
plt.plot(np.cumsum(pnl))


daily_position_all = np.array(daily_predict)*2-1
pnl = daily_position_all * spy_return
plt.plot(np.cumsum(pnl))

#print prediction accuracy
pd.crosstab(np.array(spy_sign[train_length:]), np.array(daily_predict[train_length:]), rownames=["Actual"], colnames=["Predicted"])

