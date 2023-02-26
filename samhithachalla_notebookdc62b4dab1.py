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




#reading csv as dataframes
import os
import pandas as pd
from pandas import DataFrame
os.chdir("../input/")
df_tr= DataFrame.from_csv("train.csv",header=0,sep=',',index_col='PassengerId')
df_tr.head()




#checking for a missing value in the dataframe
print(df_tr.count())
#the column age is missing 177 values
#the column Embarked is missing 2 values
#these 2 columns can be used as features
#but the column Cabin has 687 missing values and hence cant be used as a feature




#creating a data frame to fill true or false if data is present or absent correspondingly
a = pd.isnull(df_tr)
a.tail()
em_miss_index = a[a['Embarked']==True].index.values.tolist()
df_tr.loc[em_miss_index]




age_miss_index = a[a['Age']==True].index.values.tolist()
df_tr.loc[age_miss_index]




# age_miss=df_tr.iloc[age_miss_index]

