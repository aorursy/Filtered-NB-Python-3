#!/usr/bin/env python
# coding: utf-8



import pandas as pd
pd.set_option('max_rows', 5)
from learntools.core import binder; binder.bind(globals())
from learntools.pandas.creating_reading_and_writing import *
print("Setup complete.")




# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.
fruits = pd.DataFrame ([[30,21]], columns=['Apples', 'Bananas'])

q1.check()
fruits




# Uncomment the line below to see a solution
q1.solution()




# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.
fruit_sales = pd.DataFrame ([[35,21],[41,34]], columns=['Apples', 'Bananas'], index=['2017 Sales', '2018 Sales'])

q2.check()
fruit_sales




#q2.solution()




## 3.

# Create a variable `ingredients` with a `pd.Series` that looks like:

# ```
# Flour     4 cups
# Milk       1 cup
# Eggs     2 large
# Spam       1 can
# Name: Dinner, dtype: object
# ```




quantities=['4 cups', '1 cup', '2 large', '1 can']
items=['Flour', 'Milk', 'Eggs', 'Spam']
ingredients=pd.Series(quantities, index=items, name='Dinner')
q3.check()
ingredients




q3.solution()




reviews = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col=0)

q4.check()
reviews




q4.solution()




animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
animals




# Your code goes here
animals.to_csv('cows_and_goats.csv')
q5.check()




q5.solution()




import sqlite3
conn= sqlite3.connect("../input/pitchfork-data/database.sqlite")
music_reviews =pd.read_sql_query("SELECT*FROM artists", conn)

q6.check()
music_reviews




q6.solution()

