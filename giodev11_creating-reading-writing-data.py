#!/usr/bin/env python
# coding: utf-8



import pandas as pd
pd.set_option('max_rows', 5)




import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *




check_q1(pd.DataFrame())




# Your code here
data = pd.DataFrame({
    'Apples' : [30] , 
    'Bananas' : [21]
})




data




check_q1(pd.DataFrame({
    'Apples' : [30] , 
    'Bananas' : [21]
}))




# Your code here
pd.DataFrame({'Apples' : [35, 41], 'Bananas': [41, 34]}, index = ['2017 Sales', '2018 Sales'])




# Your code here
pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index= ['Flour', 'Milk', 'Eggs', 'Spam'], name = ' Dinner')




get_ipython().run_line_magic('pinfo', 'pd.read_csv')




# Your code here 
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)




# Your code here
pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheetname = 'Pregnant Women Participating')




q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])




# Your code here
q6_df.to_csv("cows_and_goats.csv")




import sqlite3
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
pd.read_sql_query("SELECT * FROM artists", conn)

