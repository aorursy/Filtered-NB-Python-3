#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data=(os.listdir("../input"))


# Any results you write to the current directory are saved as output.




data=pd.read_csv("../input/heart.csv")




data.info()




data.corr()




f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()




data.head()




data.tail()




data.shape




data.index




data.axes




data.columns




data2=data.describe()
print(data2)




print('std: ',data2.at['std','age'])




x_values=[item for item in data.age]

y_values=[item for item in data.trestbps]
   
    
plt.scatter(x_values,y_values)
plt.xlabel('Age')              # label = name of label
plt.ylabel('Resting Blood Sugar Pressure')
plt.title('Line Plot')            # title = title of plot
plt.show()




print(data['age'].value_counts(dropna =False))  #this shows how many times that age is repeated. 




data3 = data.head(3)
data3




melted = pd.melt(frame=data_new, value_vars= ['age','thalach'])
melted




melted.pivot(columns = 'variable',values='value')




data_A=data.head(3)
data_B=data.tail(3)

conc_data_row = pd.concat([data_A,data_B],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row




data.dtypes




# Data type can be converted as below:
    




data['age'] = data['age'].astype('float')
data.dtypes




print(data2)




print(data2['age'])






