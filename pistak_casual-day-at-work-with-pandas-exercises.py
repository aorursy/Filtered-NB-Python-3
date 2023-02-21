#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'




Image('http://media.indiedb.com/images/articles/1/211/210412/kickstarter-logo-whitebg.jpg')




# go for it :)
ks = #
ks.head(3)




# 1. exploratory function




# 2. exploratory function




# 3. exploratory function




# 1. visualize data




# create a deep copy of pandas data frame
clean_ks = ks.copy(deep=True)




cols_to_use = ['amt.pledged', 'by', 'category', 'currency', 'goal', 'location'
               , 'num.backers', 'num.backers.tier', 'pledge.tier', 'title']
clean_subset = clean_ks[cols_to_use]
clean_subset.head(3)




# but first, rename the columns, so instead of '.' between words in column names, we will use '_' 
# ex: 'pledge.tier' will be 'pledge_tier'
clean_subset.columns = clean_subset.columns.str.replace(# fill in just this small piece :) )
clean_subset.columns




clean_subset.head(3)




# also, rename the column 'by' to column 'author'
clean_subset = #
clean_subset.head(3)




# convert to correct datatype columns that need to (you find out which, there is one :) )
clean_subset['amt_pledged'] = clean_subset['amt_pledged'].astype('float')
type(clean_subset['amt_pledged'][0])




# create a column for average contribution by user
clean_subset['avg_contribution'] = #
clean_subset.head(3)




get_ipython().run_line_magic('pinfo2', 'pd.DataFrame.idxmax')




# there is more than 1 currency used, but we would like to make the amount comparable,
# so convert it all to the currency that is the most common
most_common_currency = #
# translate to python: group by currency, count frequency, use .idxmax() in the end
most_common_currency




most_common_currency.idxmax()




# you're all alone here :) 
# resulting data frame should contain 2 new columns: conversion rates for each currency, converted amount in USD
currencies = ['usd', 'sek', 'nzd', 'gbp', 'eur', 'dkk', 'chf', 'cad', 'aud']
conversion = [1.00, 0.13, 0.71, 1.30, 1.19, 0.16, 1.04, 0.80, 0.79]




conversion_rates = [ ('currency', currencies),
                     ('conversion_rate', conversion) ]
conversion_table = pd.DataFrame.from_items(conversion_rates)
conversion_table




get_ipython().run_line_magic('pinfo2', 'pd.merge')




clean_currencies = #
clean_currencies[clean_currencies['currency'] != 'usd'].tail(3)




clean_currencies['usd_pledged'] = #
clean_currencies.head(3)




# you're all alone here :)
# resulting data frame should contain 2 new columns: conversion rates for each currency, converted amount in USD
 # use series.map(lookup function) to series




clean_currencies['avg_contribution'] = clean_currencies['usd_pledged'] / clean_currencies['num_backers']




# get value behind the last comma in the column location and trim off the left space in the resulting string. regex somebody?
# or create a list of elements in location. Elements are split by coma
country_list = #
# strip is needed, because the country contains space on the left side of string
country_list[0][-1].strip()




clean_currencies['country'] = country_list.map(lambda x: x[-1].strip())
clean_currencies.head(3)




# Do companies also use kickstarter or only private subjects, if both, what is the ratio between them?
# oh_snap =  you can try this at home with male and female names provided
# http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/male.txt
# http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/female.txt




# count frequency of original currency used in projects
most_common_currency.sort_values(ascending=False)




# Are there people, that start multiple succesfull funds or it is not common?
# yep, we can answer this, use groupby
multiple_campaigns = 
type(multiple_campaigns)




multiple_campaigns[multiple_campaigns['author'] > 1].sum()




more_than_one = len(multiple_campaigns[multiple_campaigns['author'] > 1])
more_than_one




# Percentage of people or companies that have more than one successful campaign in top 4000 campaigns
more_than_one/len(multiple_campaigns)




461/((4000-1324)+461)




# What type of products are sold this way, what type of industry is the most successful on kickstarter 
# (fun, games, free time, culture, music, paintings, books, gadgets, etc.)
industry_frequency = #
industry_frequency




top30_industry = industry_frequency[0:30]
top30_industry




# segmentation of the market and average funding for a backer
avg_back_industry = clean_currencies[['category', 'avg_contribution']] # fill in rest
avg_back_industry




# target group that is successful in getting funds in crowdfunding campaigns 
category_sum = # fill in from different side :) .sum().sort_values(by='usd_pledged', ascending=0)
# Tip - use different aggragation function than before
category_sum




# interesting findings during the process of analysis




# count frequency of countries if you got them from the location column
ks_countries = clean_currencies['country'] #
len(ks_countries)




# create one histogram (shows distribution of 1 value and puts similar values in the same bin. 
# The more values in one bin, the bigger the bin is)
# Example: frequency of average funding size per backer in a project
clean_currencies['avg_contribution'].plot.hist(bins = 30)




# one barplot (has few categories at x axis and one value for each category at y axis)
# Example category of industry and their counts
top30_industry.plot.bar()




# one scatterplot (has x and y as numeric values and shows each observation as a point)
# Example: amount pledged vs total number of backers, or even better num_backers vs avg_contribution
contrib_scat = clean_currencies[['num_backers', 'avg_contribution']].plot.scatter(x='num_backers', y='avg_contribution',
                                                                                title='Number of backers vs average contribution')




pledge_scat = clean_currencies[['num_backers', 'usd_pledged']].plot.scatter(x='num_backers', y='usd_pledged',
                                                                                title='Number of backers vs amount pledged.')




# anything additional you like and want to share :) Boxplots are really cool!
# boxplots are like barplots that use a category and numerical value 
# Instead of just 1 number per category, boxplots use all the values to create a specific box for each category
# Or you could tell that it is a scatterplot for categories that has nice properties shown (quantiles, median, outliers)
top30_industries_list = top30_industry.index.values.tolist()
indus_box = clean_currencies[clean_currencies['category'].isin(
                    top30_industries_list)].boxplot(column='avg_contribution', by='category',
                                                    figsize=(40, 10), fontsize=10)




# Save the plot into file as png
fig = indus_box.get_figure()
fig.savefig('industry_avgcontributions_boxplot.png', bbox_inches='tight')




# Make either scatter plot or barplot more visually engagind
# use color to emphasize important information that you want to communicate to your audience
indus_colors = cm.plasma(np.linspace(0.05,0.95,len(top30_industry)))
indus_bar = top30_industry.plot.bar(figsize=(30, 10), fontsize=20, color=indus_colors, 
                                    title='Top 30 industries frequency in successful KS projects')




# Save the plot into file as png
fig = indus_bar.get_figure()
fig.savefig('industry_frequency_barplot.png', bbox_inches='tight')




# to save to excel is as easy as to csv with the difference, that you can use multiple sheets in one file
writer = pd.ExcelWriter('Kickstarter_analysis.xlsx')
# write different dataframes to multiple sheets here :)
writer.save()




category_sum.to_csv('Kickstarter_category_sum.csv', sep=';')

