#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
from sklearn.model_selection import train_test_split 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




randomGen = np.random.RandomState(1)
X1 = [randomGen.randint(500,2000) for a in randomGen.rand(50)]
X2 = [randomGen.randint(100,500) for b in randomGen.rand(50)]
X3 =  pd.Series(X1) * 3 + 12
Y = pd.Series(X2) + pd.Series(X3) 

data = pd.DataFrame({
    'input X1':X1,
    'input X2':X2,
    'input X3':X3,
    'output':Y
})
dataX1 =pd.DataFrame({
    'input X1':X1,
    'output':Y
})
dataX2 =pd.DataFrame({
    'input X2':X2,
    'output':Y
})
dataX3 =pd.DataFrame({
    'input X3':X3,
    'output':Y
})
dataOut =pd.DataFrame({
    'output Y':Y
})




data




data.corr()




#X1 and Y
print('Correlation between X1 and Y')
dataX1.corr()




#X2 and Y
print('Correlation between X2 and Y')
dataX2.corr()




#X3 and Y
print('Correlation between X3 and Y')
dataX3.corr()




#X1 and Y
data.plot(kind='scatter',x='input X1',y='output',
           title="Scatter plot showing relationship between X1 and Y",
            figsize=(10,8))





data.plot(kind='scatter',x='input X2',y='output',
           title="Scatter plot showing relationship between X2 and Y",
            figsize=(10,8))




from sklearn.model_selection import train_test_split 
from sklearn import linear_model
import statsmodels.api as sm




dataReg = data.copy()
Regres = linear_model.LinearRegression()

print ('separation of data into dependent (Y) and independent(X1 and X2) variables')
X_data = dataReg[['input X1']]#independent
X2_data = dataReg[['input X2']]#independent
Y_data = dataReg['output']#dependent
Y2_data = dataReg['output']#dependent

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
X2_train,X2_test, y2_train,y2_test = train_test_split(X2_data, Y2_data, test_size=0.30)
pd.DataFrame(X_test)




model = Regres.fit(X_train,y_train)
model2 = Regres.fit(X2_train,y2_train)
predictions= Regres.predict(pd.DataFrame(X_test))
predictions2 = Regres.predict(pd.DataFrame(X2_test))




plt.scatter(X_test,predictions, color='black')
plt.show()




plt.scatter(X2_test,predictions2, color='blue')
plt.show()




import seaborn as sns




# Plot the residuals after fitting a linear model
sns.residplot(Regres.predict(X_train), Regres.predict(X_train)-y_train, lowess=True, color="r")
sns.residplot(Regres.predict(pd.DataFrame(X_test)), Regres.predict(pd.DataFrame(X_test))-y_test, lowess=True, color="g")
plt.title('Residual Plot using Training (red) and test (green) data ')
plt.ylabel('Residuals')




from sklearn.metrics import r2_score




print('Coefficicent is: %.2f' % r2_score(y_test, Regres.predict(X_test)))




Regres.intercept_




Regres= linear_model.LinearRegression()




print(model.score(pd.DataFrame(X_test), pd.DataFrame(y_test)))
print(model.score(pd.DataFrame(X2_test), pd.DataFrame(y2_test)))




#R^2
Regres(X2_test,y_test) 






