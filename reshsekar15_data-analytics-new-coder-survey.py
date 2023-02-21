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




data=pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv")




data.columns.values




# Looking out for missing values and removing unwanted features




def feature_summary(data):
    n_row=data.shape[0]
    features=pd.DataFrame()
    features_names=[]
    features_counts=[]
    features_missing=[]
    names=data.columns.values
    for i in names:
        features_names.append(i)
        features_counts.append(data[i].value_counts().count())
        features_missing.append(data[data[i].isnull()].shape[0])
    features['name']=features_names
    features['value counts']=features_counts
    features['missing']=features_missing
    features['percentage_missing']=features['missing']/n_row
    return (features)
        




feature_table=feature_summary(data)




feature_table




useful_columns=feature_table[feature_table['percentage_missing']<0.50]
# eliminating features with more than 50% missing or null values
useful_columns.shape









import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data_IsSoftwareDev = data[data['IsSoftwareDev']==1]
data_IsSoftwareDev['Age'].sort_values(axis=0,ascending=False).value_counts().plot(kind='bar',figsize=(12,4))
# this could be skewed based on number of people who took the survey
# let us divide the series  by the total number of value counts and then plot a bar plot






