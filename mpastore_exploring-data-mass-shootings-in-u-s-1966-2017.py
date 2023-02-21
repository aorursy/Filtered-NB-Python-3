#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




# Read in dataset (Ver 5)
df = pd.read_csv("../input/Mass Shootings Dataset Ver 5.csv", encoding = "ISO-8859-1", parse_dates=["Date"])




# Peek into dataset to see what we're working with
df.head()




# Delete unnecessary 'S#' column, and also remove lat/long (won't be using them)
df.drop(['S#', 'Latitude', 'Longitude'], axis=1, inplace=True)
df.head()




# Rename columns for easier usability
new_cols = ['title', 'location', 'date', 'incident_area', 'open_closed_location', 'target',             'cause', 'summary', 'num_fatalities', 'num_injured', 'num_total_victims', 'num_police_killed',             'age', 'employed', 'employed_at', 'mental_health', 'race', 'gender']
df.columns = new_cols
print(df.columns)




# I want to index these by datetime, but there are some duplicates...
date_sort = df.set_index('date', drop=True, verify_integrity=True)




# Reorder dataframe by date, reset index
df = df.sort_values('date').reset_index(drop=True)
df.head()




# Drop some other columns that are full of nulls, since we won't use those either
df.drop(['open_closed_location', 'employed', 'employed_at'], axis=1, inplace=True)
df.info()




# Standardize race column
df.race.value_counts()




# Condense all null into "Unknown"
df.race.fillna('Unknown', inplace=True)

# Condense all "Other"
df.race = df.race.apply(lambda x: x.replace('Some other race', 'Other'))

# Condense all "White"
df.race = df.race.apply(lambda x: x.replace('White American or European American', 'White'))
df.race = df.race.apply(lambda x: x.replace('white', 'White'))

# Condense all "Black"
df.race = df.race.apply(lambda x: x.replace('Black American or African American', 'Black'))
df.race = df.race.apply(lambda x: x.replace('black', 'Black'))

# Condense all "Asian"
df.race = df.race.apply(lambda x: x.replace('Asian American', 'Asian'))

# Change "/" to " & " to better communicate meaning (multiple individuals of
# separate races involved in incident)
df.race = df.race.apply(lambda x: x.replace('/', ' & '))

df.race.value_counts()




# Standardize gender column
df.gender.value_counts()




# Collapse "M"
df.gender = df.gender.apply(lambda x: x.replace('Male', 'M'))

# Collapse "F"
df.gender = df.gender.apply(lambda x:x.replace('Female', 'F'))

# Collapse "M & F"
df.gender = df.gender.apply(lambda x: x.replace('Male/Female', 'M & F'))
df.gender = df.gender.apply(lambda x: x.replace('M/F', 'M & F'))

df.gender.value_counts()




# Create version without Las Vegas shooting for more readable graphs (since it's such an outlier)
_ = df[df.num_total_victims > 500] # I used this to find index: 319
sans_vegas = df.drop(319)




get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')




sns.set()

# View swarm plot of total victims by race (without Vegas outlier)
plt.figure(figsize=(14,8))
_ = sns.swarmplot(x='race', y='num_total_victims', data=sans_vegas, hue='gender')
_ = plt.xlabel('Race')
_ = plt.ylabel('# Total Victims')
_ = plt.xticks(rotation=80)
plt.show()




# View swarm plot of total fatalities by race (including Vegas outlier)
plt.figure(figsize=(14,8))
_ = sns.swarmplot(x='race', y='num_fatalities', data=df, hue='gender')
_ = plt.xlabel('Race')
_ = plt.ylabel('# Fatalities')
_ = plt.xticks(rotation=80)
plt.show()




# Define ECDF function
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y




# ECDF for fatalities, injured, and total # victims (remove Vegas outlier for graph readability)
x, y = ecdf(sans_vegas.num_total_victims)

plt.figure(figsize=(14,8))
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Total # of Victims')
_ = plt.ylabel('ECDF')
_ = plt.xticks(np.arange(125, step=10))

# Display the plot
plt.show()




# Number of mass shootings by year
year_shootings = df.date.dt.year.value_counts()

plt.figure(figsize=(14,8))
sns.barplot(year_shootings.index, year_shootings.values, alpha=0.8, color='red')
_ = plt.xlabel('Year', fontsize=20)
_ = plt.ylabel('# of Mass Shootings', fontsize=20)
_ = plt.xticks(rotation=50)
plt.show()




# Number of mass shootings by month
month_shootings = df.date.dt.month.value_counts()

plt.figure(figsize=(14,8))
sns.barplot(month_shootings.index, month_shootings.values, alpha=0.8, color='red')
_ = plt.xlabel('Month', fontsize=20)
_ = plt.ylabel('# of Mass Shootings', fontsize=20)
plt.show()




# Which months have the deadliest shootings?
plt.figure(figsize=(14,8))
sns.boxplot(sans_vegas.date.dt.month, sans_vegas.num_total_victims, palette='deep')
_ = plt.xlabel('Month', fontsize=20)
_ = plt.ylabel('# of Mass Shootings', fontsize=20)
_ = plt.yticks(np.arange(110, step=10))
plt.show()




# Use data from the last 5 years only (since there's been such an uptick in # of shootings)
lastfive = df[df.date.dt.year >= 2012]
lastfive.info()




lastfive.reset_index(drop=True, inplace=True)
lastfive.head()






