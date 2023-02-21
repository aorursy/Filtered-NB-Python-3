#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




filename = check_output(["ls", "../input"]).decode("utf8").strip()
print(filename)




df = pd.read_csv("../input/" + filename, engine = 'python', sep = ',')




df.head()
del df['user_id']




np.mean(df['scala'] > 0)




df.describe()




dk = df > 0
dk.head()




knownLangs = np.mean(dk)




knownLangs.sort_values(ascending = False)




javascript           0.796461 
python               0.459481 
java                 0.336521 
c                    0.247752 
c++                  0.221007




firstLangs = df.idxmax(axis = 1, skipna= True)




# What are the most popular "first" language. (find max in each row)
firstLangs.value_counts()/len(df)




javascript          0.342478
python              0.116946 
java                0.066777 
go                  0.035164  
c++                 0.017296 
c                   0.013459 
scala               0.008476 
haskell             0.005154 











