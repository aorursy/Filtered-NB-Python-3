#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

summer = pd.read_csv('../input/summer.csv')
winter = pd.read_csv('../input/winter.csv')




summer.head()




winter.head()




print(summer.shape)
print(winter.shape)




#Find missing values by percent
def percent_missing_columnwise(data):
   notnullcount = data.count()
   percent_values = [(1 - col/data.shape[0])*100 for col in notnullcount]
   return pd.DataFrame(percent_values, index=data.columns)




# Summer missing values - columnwise percentage
percent_missing_columnwise(summer)




# Only the country column has some missing values
# Delete those rows




indices_to_delete = summer[summer['Country'].isnull()].index




summer = summer.drop(indices_to_delete)




get_ipython().run_line_magic('pinfo', 'pd.to_datetime')




percent_missing_columnwise(winter)




# Winter has no missing values




# Let's convert the categorical variable to numeric using LabelEncoder

from sklearn.preprocessing import LabelEncoder

summer_updated = summer.loc[:,'City':'Medal'].apply(LabelEncoder().fit_transform)




summer_updated.loc[:,'Year'] = summer['Year']




winter_updated = winter.loc[:,'City':'Medal'].apply(LabelEncoder().fit_transform)




winter_updated.loc[:,'Year'] = winter['Year']




winter_updated.head()






