#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar, datetime

sns.set(style="whitegrid", palette="Set2")

CITY = 'Torun'
SELECTED_CITIES = ['Torun', 'Cracow', 'Katowice', 'Olsztyn', 'Szczecin', 'Bialystok',
                   'Zielona Gora', 'Warsaw', 'Poznan', 'Bialystok', 'Wroclaw']


START_YEAR = 1753
MONTH = int(datetime.datetime.now().strftime("%m"))
# when global average temperature started to rise
RISE_START_YEAR = 1970

# Baseline years
BASE_YEAR_START = 1850
BASE_YEAR_END = 1900




df = pd.read_csv('../input/GlobalLandTemperaturesByCity.csv')

date_index = pd.to_datetime(df['dt'], format='%Y-%m-%d', errors='ignore')

df['di'] = date_index
df['Year'] = df['di'].dt.year
df['Month'] = df['di'].dt.month
df['Day'] = df['di'].dt.day

poland = df[df['Country'] == 'Poland'].copy()
poland_clean = poland.copy()

poland_clean.drop('Latitude', axis=1, inplace=True)
poland_clean.drop('Longitude', axis=1, inplace=True)
poland_clean.drop('AverageTemperatureUncertainty', axis=1, inplace=True)
poland_clean.drop('Country', axis=1, inplace=True)
poland_clean.drop('dt', axis=1, inplace=True)

city = df[df['City'] == CITY].dropna().copy()
climate_city = city[city['Year'] >= START_YEAR].copy()
month_city = climate_city[climate_city['Month'] == MONTH].copy()
current_month_city = month_city[month_city['Year'] > RISE_START_YEAR]
baseline_month_city = month_city[
    month_city['Year'].between(BASE_YEAR_START, BASE_YEAR_END)].copy()




poland['City'].unique()




sns.lmplot(x='Year', y='AverageTemperature', hue='City', data=poland[poland['City'].isin(SELECTED_CITIES)], 
           scatter=False, lowess=True)




climate_city.head()




print("Average temperature in {city} in {from_year}-2013: {temperature}".format(
    city=calendar.month_name[MONTH],
    from_year=START_YEAR,
    temperature=month_city['AverageTemperature'].mean()
))
print("Average temperature in {city} in {from_year}-2013: {temperature}".format(
    city=calendar.month_name[MONTH],
    from_year=RISE_START_YEAR,
    temperature=current_month_city['AverageTemperature'].mean()
))
print("Average temperature in {city} in {from_year}-{to_year}: {temperature} (baseline)".format(
    city=calendar.month_name[MONTH],
    from_year=BASE_YEAR_START,
    to_year=BASE_YEAR_END,
    temperature=baseline_month_city['AverageTemperature'].mean()
))




sns.swarmplot(x='Month', y='AverageTemperature', data=climate_city, size=1, palette='Set2')




from scipy.stats import spearmanr

sns.jointplot(month_city['Year'], month_city['AverageTemperature'], 
              kind="hex", stat_func=spearmanr, color="#4CB391")




before_1970 = month_city[month_city['Year'] < RISE_START_YEAR]['AverageTemperature'].copy()
after_1970 = month_city[month_city['Year'] >= RISE_START_YEAR]['AverageTemperature'].copy()
after_1995 = month_city[month_city['Year'] >= 1995]['AverageTemperature'].copy()




sns.distplot(before_1970, hist=False)
sns.distplot(after_1970, hist=False)
sns.distplot(after_1995, hist=False)




def label_bin(row):
    if row['AverageTemperature'] < (row['MonthMeanBaseline'] - row['MonthStdBaseline']):
        return 'Low'
    elif row['AverageTemperature'] > (row['MonthMeanBaseline'] + row['MonthStdBaseline']):
        return 'High'
    return 'Medium'




monthly_frames = []
for i, group in climate_city.groupby('Month'):
    g2 = group.copy()
    g_temp_series = g2[(g2['Year'] >= BASE_YEAR_START) & (g2['Year'] <= BASE_YEAR_END)]['AverageTemperature']
    g2['MonthMeanBaseline'] = g_temp_series.mean()
    g2['MonthStdBaseline'] = g_temp_series.std()
    g2['TemperatureBin'] = g2.apply(label_bin, axis=1)
    monthly_frames.append(g2)




all_with_bins = pd.concat(monthly_frames).sort_index()
city_grouped_with_counted_bins = all_with_bins.groupby(['Year','TemperatureBin'])['City']     .agg(['count']).reset_index()




sns.lmplot(x='Year', y='count', hue='TemperatureBin', 
           data=city_grouped_with_counted_bins, 
           scatter=False, lowess=True)

