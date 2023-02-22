#!/usr/bin/env python
# coding: utf-8



## imports
import pandas as pd
import numpy as np
import os
import datetime
import seaborn as sns




# read in all three datasets
recipes = pd.read_csv("../input/epirecipes/epi_r.csv")
bikes = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")
weather = pd.read_csv("../input/szeged-weather/weatherHistory.csv")




print("Recipes dataset with dimensions ",recipes.shape )
col_names_recipes = recipes.columns
#for i in col_names_recipes:
#    print(i, end=' , ')




print("Bikes dataset with dimensions ",bikes.shape )
col_names_bikes = bikes.columns
print("The columns of bikes are:\n")
for i in col_names_bikes:
    print(i, end=' , ')
print("The first 3 rows of dataset:\n")
print(bikes.head(3))

print("\n\nWeather dataset with dimensions ",weather.shape )
col_names_weather = weather.columns
print("The columns of weather are:\n")
for i in col_names_weather:
    print(i, end=' , ')
print("The first 3 rows of dataset:\n")
print(weather.head(3))




print("\n Changing the data column to have just date")
#print(bikes.dtypes)
#print(weather.dtypes)
bikes.columns = [c.strip(' ').replace(' ', '_') for c in bikes.columns]
weather.columns = [c.strip(' ').replace(' ', '_') for c in weather.columns]

temp = pd.DataFrame(bikes.Date.str.split(' ').tolist(),  columns = ['Date','extra'])
bikes['Date'] = temp['Date']

temp = pd.DataFrame(weather.Formatted_Date.str.split(' ').tolist(),  columns = ['Date','x','y'])
weather['Just_Date'] = temp['Date']




# removing rows with NULL for both bikes and weather dataset
print("bikes size before",bikes.shape)
bikes.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print("bikes size after",bikes.shape)
print("weather size before",weather.shape)
weather.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print("weather size after",weather.shape)




print("Columns in weather data :",weather.columns,"\n\n weather data set correlation matrix.\n" )
corr = weather.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, 
           center = 0, cmap="YlGnBu")




# are the ratings all numeric?
print("Is this variable numeric?")





# are the ratings all integers?
print("Is this variable only integers?")





# plot calories by whether or not it's a dessert





# plot & add a regression line


# your work goes here! :)

