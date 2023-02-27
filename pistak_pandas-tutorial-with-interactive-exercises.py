#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import pyplot as plt




get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'




Image('http://cdn-static.denofgeek.com/sites/denofgeek/files/pokemon_4.jpg')




# prints current working directory full path
os.getcwd()




# reads csv file into 
poke = pd.read_csv('../input/Pokemon.csv')




# shows first  few rows of table
poke.head()




# data frame
type(poke)




# series
type(poke['Name'])




# numpy array
type(poke['Name'].values)




# basic python/numpy data type - str, int, float...
type(poke['Name'].values[0])




poke['Name'].head()




len(poke)




len(poke) == len(poke['Name'])




# general information about columns
poke.info()




# summary statistics of numeric columns
poke.describe()




# How many null values are there?
poke.isnull().sum()




# How many legendary and common pokemon are there?
poke['Legendary'].sum()




# boolean filtering
poke[poke['Name'] == 'Pikachu']




# subsetting with .loc and .iloc
poke[poke.loc[:,'Name'].isin(['Pikachu', 'Bulbasaur', 'Charmander', 'Squirtle'])]




# creating subset data frame
image_poke = poke[poke.loc[:,'Name'].isin(['Pikachu', 'Bulbasaur', 'Charmander', 'Squirtle'])]
image_poke




# how to make plot with pandas and what arguments to pass?
get_ipython().run_line_magic('pinfo2', 'image_poke.plot')
# two question marsk show full docstring that is present at documentation webpage for pandas.




# this draws it in a cell
# barplot that compares attack of chosen pokemon 
image_poke.plot.bar(x='Name', y='Attack', color=['green', 'red', 'blue', 'yellow'], title='Attack Comparison')




poke = poke.rename(columns={'#':'Number'})
poke.columns




# Only common means non-legendary :)




# Finish subset of DataFrame using condition so that only pokemon that don't have Mega in their name are selected (hint: ~)




# check if there is only 1 pokemon for every number
poke['Number'].groupby(poke['Number']).count().sort_values(ascending=False)
# we don't want to show so long output afterwards, so we can just add ';' behind the command not to show the output afterwards.




# Finish the subset




# Reindex the dataframe, so the first index is 0 and the last is n-1




# How many unique pokemon are there in the remaining dataset?
len(nodup_poke['Number'].unique())




# Which pokemon type is the strongest and which the weakest on average? (according to total stats)
strongest_type_avg = nodup_poke[['Type 1','Total']].groupby(nodup_poke['Type 1']).mean().sort_values(by='Total', ascending=False)
strongest_type_avg




# Find the strongest pokemon in each group based on Type 1 and order them alphabetically from A-Z.
# Notice, this one has different logic - we do sorting first and only then group by and take the first observation for every group.
strongest_type_max = nodup_poke[['Total', 'Name', 'Type 1']].sort_values(
                                by='Total', ascending=False).groupby('Type 1', as_index=False).first()
strongest_type_max 




# Which pokemon type is the most frequent?
type_frequency = nodup_poke['Type 1'].# fill in the rest, sort from largest to smallest
type_frequency




# What are the 5 strongest pokemon among the common pokemon?
top5_poke = nodup_poke[['Name', 'Total']].sort_values(by='Total', ascending=False)[0:5]
top5_poke




# Which pokemon generation has the biggest average total stats?




# What type is pikachu?
pikachu_type = nodup_poke['Type 1'][nodup_poke['Name'] == 'Pikachu'].values[0]
pikachu_type




# How strong is Pikachu among pokemon of the same type? Hint: debugging
pikachu_rank = nodup_poke[nodup_poke['Type 1'] == pikachu_type].sort_values();
pikachu_rank




# reset index of table to start from 0 to n-1
pikachu_rank.reset_index(inplace=True)
pikachu_rank




# Create a histogram of all common pokemon's total stats
# fill in missing .plot.hist(bins=15)




import matplotlib.cm as cm  # these would normally be at the beginning of the notebook

# colormaps https://matplotlib.org/examples/color/colormaps_reference.html
type_colors = cm.spring(np.linspace(0.05,0.95,len(type_frequency)))
type_frequency.plot.bar(color=type_colors)




# Create a boxplot of average total stats by type. Hint: use tables we created in previous step
nodup_poke[nodup_poke['Type 1'].isin(['Bug', 'Water', 'Grass', 'Poison'])
          ].boxplot(column='Total', by='Type 1')




# Create a boxplot of total stats by generation
# fill in missing .boxplot(column='Total', by='Generation')




# Create a barplot of total stats by generation
generation_comparison['Generation'] = generation_comparison['Generation'].astype('str')
gen_colors = cm.summer(np.linspace(0.05,0.95,len(generation_comparison)))
generation_comparison.plot.bar(x='Generation',color=gen_colors, title='Comparison of avg total stats between pokemon generations')




# Create a barplot of strongest pokemon in each Type 1. One color for all is fine.
# fill in all :) 




# Show Pikachu's total stats among other pokemon of the same type together with generation of pokemon
pikachu_rank['color'] = 'Grey'
pikachu_rank['size'] = 30
pikachu_rank['color'][pikachu_rank['Name']=='Pikachu']='Yellow'
pikachu_rank['size'][pikachu_rank['Name']=='Pikachu']=100
pika_plot = pikachu_rank.plot.scatter(x='Generation', y='Total', 
                                      c=pikachu_rank['color'], s=pikachu_rank['size'],
                          title='Pikachu vs other electric pokemon')




fig = pika_plot.get_figure()
fig.savefig('pika_plot.png')

