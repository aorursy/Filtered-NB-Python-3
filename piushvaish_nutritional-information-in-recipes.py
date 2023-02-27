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




import numpy as np 
import pandas as pd 
pd.options.display.max_columns = 100
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
warnings.filterwarnings('ignore') 




train = pd.read_csv("../input/epi_r.csv")




#Pandas allows you to have a sneak peak at your data.
train.head(5)




df = train.iloc[:,0:6]




df.head(2)




import re
missing_values = []
nonumeric_values = []

print ("TRAINING SET INFORMATION")
print ("========================\n")

for column in train:
    # Find all the unique feature values
    uniq = train[column].unique()
    print ("'{}' has {} unique values" .format(column,uniq.size))
    if (uniq.size > 10):
        print("~~Listing up to 10 unique values~~")
    print (uniq[0:10])
    print ("\n-----------------------------------------------------------------------\n")
    
    # Find features with missing values
    if (True in pd.isnull(uniq)):
        s = "{} has {} missing" .format(column, pd.isnull(train[column]).sum())
        missing_values.append(s)
    
    # Find features with non-numeric values
    for i in range (1, np.prod(uniq.shape)):
        if (re.match('nan', str(uniq[i]))):
            break
        if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(uniq[i]))):
            nonumeric_values.append(column)
            break
  
print ("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print ("Features with missing values:\n{}\n\n" .format(missing_values))




# Calories, protein, fat and sodium have almost same missing values. Let us explore these.




#Check when a value in a cloumn is NaN
caloriesNan = df[pd.isnull(df).any(axis=1)]




caloriesNan




sns.kdeplot(df['calories'], shade=True)




df.plot(kind="scatter",     # Create a scatterplot
              x= "protein",          # Put protein on the x axis
              y= "calories")         # Put calories on the y axis




ax = sns.regplot(x="fat", y="calories", data= df,x_jitter=.6)




sns.lmplot('fat', 'calories', 
           data=df, 
           fit_reg=False, 
           #dropna=True,
           hue="rating",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Scatterplot of fat, calories vs rating')
plt.xlabel('Fat')
plt.ylabel('Calories')




sns.lmplot('protein', 'calories', 
           data=df, 
           fit_reg=False, 
           #dropna=True,
           hue="rating",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Scatterplot of Protein, calories vs rating')
plt.xlabel('Protein')
plt.ylabel('Calories')




sns.lmplot('fat', 'sodium', 
           data=df, 
           fit_reg=False, 
           palette="Set1",           
           hue="rating",
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Scatterplot of Fat, Sodium vs rating')
plt.xlabel('Fat')
plt.ylabel('Sodium')




df[(df['sodium'] > 6252)].count()




print(df.skew())




# **Will there be any correlation between Nutrition features**




corr = df.iloc[:, 1:6].corr()
plt.figure(figsize=(5, 5))
sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')




cor_dict = corr['protein'].to_dict()
del cor_dict['protein']
print("List the numerical features decendingly by their correlation with Protein:\n")
for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
    print("{0}: \t{1}".format(*ele))

