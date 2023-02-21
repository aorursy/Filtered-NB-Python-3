#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1], [np.nan, np.nan, np.nan, 5],[3, 4, np.nan, 1], [3, 4, 0, 1]], columns=list('ABCD'))




df.shape




df




df.drop_duplicates() #It will remove index 3 since it is dublicate to 1




get_ipython().run_line_magic('pinfo', 'df.dropna')




df.dropna(axis=1, how='all') #remve all column where all value is 'NaN' exists




df.dropna(axis=0, how='all') #remve all row where all value is 'NaN' exists




df.dropna(axis=1, how='any') #remve all column where any value is 'NaN' exists




df.dropna(axis=0, how='any') #remve all column where all value is 'NaN' exists




df.dropna(thresh=2) #remve all row if there is non-'NaN' value is less than 2




df.dropna(axis=1, thresh=2) #remve all column if there is non-'NaN' value is less than 2




df.dropna(axis=0, subset=['A']) #remove row where if there is any 'NaN' value in column 'A'




df.dropna(axis=1, subset=[1]) #remove column  if there is any 'NaN' value in index is '1'






