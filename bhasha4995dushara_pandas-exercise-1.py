#!/usr/bin/env python
# coding: utf-8



import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *




check_q1(pd.DataFrame())




df = pd.DataFrame({'Apples':[30],'Bananas':[21]})




df = pd.DataFrame(data={'Apples':[35,41],'Bananas':[21,34]},index=['2017Sales','2018Sales'])




# **Exercise 3**: Create a `Series` that looks like this:

# ```
# Flour     4 cups
# Milk       1 cup
# Eggs     2 large
# Spam       1 can
# Name: Dinner, dtype: object
# ```




pd.Series(data=['4 cups','1 cup','2 large','1 can'],index=['Flour','Milk','Eggs','Spam'],name='Dinner')




red = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
red




e = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
e




q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])




q6_df.to_csv('cows_and_goats.csv')




import sqlite3
con = sqlite3.connect('../input/pitchfork-data/database.sqlite')
d = pd.read_sql_query('select * from artists',con)
d

