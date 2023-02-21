#!/usr/bin/env python
# coding: utf-8



# Import the required packages
get_ipython().system('pip install pywaffle')
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from pywaffle import Waffle
import geopandas as gpd




# Importing static dataset from .csv file
raw_data = pd.read_csv('/kaggle/input/vehicle-collisions/database.csv')

print(raw_data.shape)
raw_data.head(3)




# Removing columns that have more than a third values as NaN
mask = raw_data.isna().sum() / len(raw_data) < 0.34
raw_data = raw_data.loc[:, mask]

#Removing columns that don't have a large contributing factor to EDA and Predictions
cols_to_drop = ['ZIP CODE', 'LOCATION', 
                'VEHICLE 2 FACTOR', 'VEHICLE 2 TYPE']
raw_data.drop(cols_to_drop, axis = 1, inplace = True)




# Concatenating date and time columns
raw_data['CRASH_DATE_TIME'] = raw_data['DATE'] + ' ' + raw_data['TIME']

# Drop redundant date and time columns
cols_to_drop = ['DATE', 'TIME']
raw_data.drop(cols_to_drop, axis = 1, inplace = True)




# Convert Crash Date and time to datetime format
raw_data['CRASH_DATE_TIME']= pd.to_datetime(raw_data['CRASH_DATE_TIME'], 
                                            dayfirst=True, errors='coerce')

# Dropping rows with problematics dates
idx = raw_data[raw_data['CRASH_DATE_TIME'].isnull()].index
raw_data.drop(idx, axis = 0, inplace = True)

print(raw_data.shape)
raw_data.head(3)




borough_wise = raw_data.groupby(['BOROUGH']).size().reset_index(name='NoOfAccidents')
borough_wise.head()




# Import the ShapeFile for Borough Boundaries
fp = '/kaggle/input/nyc-borough-boundaries/geo_export_87071461-9196-46f3-8d1b-52fed88fb835.shp'
borough_geo = gpd.read_file(fp)
borough_geo['boro_name'] = borough_geo['boro_name'].str.upper() 

# Merging ShapeFile with data
borough_wise = borough_geo.set_index('boro_name').join(borough_wise.set_index('BOROUGH'))




# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(10, 7))

# Drawing the Map
borough_wise.plot(column = 'NoOfAccidents', cmap = 'Reds', linewidth = 0.8, 
                      ax = ax, edgecolor = '0.8')

# Map customizations
ax.axis('off')
ax.set_title('Motor Vehicle Collisions in NYC', size = 16)
ax.annotate('Source: NYC Open Data', xy = (0.1, .08),  
            xycoords = 'figure fraction', horizontalalignment = 'left', verticalalignment = 'top', 
            fontsize = 12, color = '#555555')

# Adding a color bar legend to the map
sm = plt.cm.ScalarMappable(cmap = 'Reds', 
                           norm = plt.Normalize(vmin = 22822, vmax = 189648))
sm._A = []
cbar = fig.colorbar(sm)

# Export map
fig.savefig('borough_wise_accidents.png', dpi=300)




injuries_and_fatalities = raw_data.groupby(['BOROUGH'])['PERSONS KILLED', 'PERSONS INJURED'].agg('sum').reset_index()

injuries_and_fatalities['Total Accidents'] = raw_data.groupby(['BOROUGH']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
injuries_and_fatalities['Injury%'] = round((injuries_and_fatalities['PERSONS INJURED']/ injuries_and_fatalities['Total Accidents'] * 100), 1)
injuries_and_fatalities['Fatality%'] = round((injuries_and_fatalities['PERSONS KILLED']/ injuries_and_fatalities['Total Accidents'] * 100), 3)

injuries_and_fatalities.head()




# Dropping redudant column and merging with ShapeFile
injuries_and_fatalities.drop('Total Accidents', axis = 1, inplace = True)
injuries_and_fatalities = borough_geo.set_index('boro_name').join(injuries_and_fatalities.set_index('BOROUGH'))




# Create figure and axes for Matplotlib
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 7))

# Drawing the maps
injuries_and_fatalities.plot(column = 'PERSONS INJURED', cmap = 'PuRd', linewidth = 0.8, 
                      ax = ax1, edgecolor = '0.8')
injuries_and_fatalities.plot(column = 'PERSONS KILLED', cmap = 'Reds', linewidth = 0.8, 
                      ax = ax2, edgecolor = '0.8')
injuries_and_fatalities.plot(column = 'Injury%', cmap = 'PuRd', linewidth = 0.8, 
                      ax = ax3, edgecolor = '0.8')
injuries_and_fatalities.plot(column = 'Fatality%', cmap = 'Reds', linewidth = 0.8, 
                      ax = ax4, edgecolor = '0.8')

# Map customizations
ax1.axis('off'); ax2.axis('off'); ax3.axis('off'); ax4.axis('off')
ax1.set_title('Total number of people injured in NYC', size = 10)
ax2.set_title('Total number of people killed in NYC', size = 10)
ax3.set_title('Percentage of people injured in vehicle collisions', size = 9)
ax4.set_title('Percentage of people killed in vehicle collisions', size = 9)

# Adding color bar legends to the maps
sm = plt.cm.ScalarMappable(cmap = 'PuRd', norm = plt.Normalize(vmin = 5800, vmax = 53000))
sm._A = []
cbar = fig.colorbar(sm, ax = ax1)

sm = plt.cm.ScalarMappable(cmap = 'Reds', norm = plt.Normalize(vmin = 30, vmax = 210))
sm._A = []
cbar = fig.colorbar(sm, ax = ax2)

sm = plt.cm.ScalarMappable(cmap = 'PuRd', norm = plt.Normalize(vmin = 15, vmax = 30))
sm._A = []
cbar = fig.colorbar(sm, ax = ax3)

sm = plt.cm.ScalarMappable(cmap = 'Reds', norm = plt.Normalize(vmin = 0.05, vmax = 0.015))
sm._A = []
cbar = fig.colorbar(sm, ax = ax4)

# Export map
fig.savefig('borough_wise_injury_percentage.png', dpi=500)




# Calculate the number of people killed, injured and total accidents for each contributing factor
factor_wise = raw_data.groupby(['VEHICLE 1 FACTOR'])['PERSONS KILLED', 'PERSONS INJURED'].agg('sum').reset_index()

factor_wise['Total Accidents'] = raw_data.groupby(['VEHICLE 1 FACTOR']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# 'Unspecified' factor is the most common factor in motor vehicle collissions reported by the NYPD, 
# but since these don't give us any concrete analysis, we won't consider this, and hence we drop it.
factor_wise = factor_wise.sort_values('Total Accidents', ascending = False).head(10).iloc[1:]

# Injuries and Fatalities as Percentages
factor_wise['Injury%'] = round((factor_wise['PERSONS INJURED']/factor_wise['Total Accidents'] * 100), 1)
factor_wise['Fatality%'] = round((factor_wise['PERSONS KILLED']/factor_wise['Total Accidents'] * 100), 3)

# Drop last two
factor_wise = factor_wise[:-1]
factor_wise.head(3)




factor_accidents = factor_wise.sort_values('Total Accidents', ascending = False).head(10)
factor_accidents.head(3)




# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

# Creating the plot
factor_accidents.plot(x = 'VEHICLE 1 FACTOR', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Factors causing the most number of accidents', size = 12)
ax.set_xlabel('Contributing Factor', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 20)

# Exporting the visualisation
fig.savefig('factor_accidents.png', dpi=500)

waf_df = factor_accidents[['VEHICLE 1 FACTOR', 'Total Accidents']].set_index('VEHICLE 1 FACTOR')


# Waffle Chart 
waf = plt.figure(
    FigureClass = Waffle, 
    rows = 5, 
    values = ((waf_df['Total Accidents'] / waf_df['Total Accidents'].sum()) * 100) ,
    title={'label': 'Factors causing the most number of accidents', 
           'loc': 'center', 'size': 22},
    labels=["{0} ({1}%)".format(k, round((v / waf_df['Total Accidents'].sum()) * 100), 2) for k, v in waf_df['Total Accidents'].items()],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.15), 'ncol': len(waf_df), 'framealpha': 0},
    starting_location='NW',
    figsize=(22, 8)
)

waf.gca().set_facecolor('#EEEEEE')
waf.set_facecolor('#EEEEEE')

# Exporting the visualisation
waf.savefig('factor_accidents_waffle.png', dpi=500)




factor_inj_rate = factor_wise.sort_values('Injury%', ascending = False).head(10)
factor_fat_rate = factor_wise.sort_values('Fatality%', ascending = False).head(10)




# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
color2 = cm.autumn(np.linspace(.2,.6, 10))

# Creating the plots
factor_inj_rate.plot(x = 'VEHICLE 1 FACTOR', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

factor_fat_rate.plot(x = 'VEHICLE 1 FACTOR', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Factors with the highest rate of injury', size = 12)
ax1.set_xlabel('Contributing Factor', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Factors with the highest rate of fatality', size = 12)
ax2.set_xlabel('Contributing Factor', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('factor_inj_fat_rate.png', dpi=500)




# Calculate the number of people killed, injured and total accidents for each vehicle type
vehicle_wise = raw_data.groupby(['VEHICLE 1 TYPE'])['PERSONS KILLED', 'PERSONS INJURED'].agg('sum').reset_index()

vehicle_wise['Total Accidents'] = raw_data.groupby(['VEHICLE 1 TYPE']).size().reset_index(name='NoOfAccidents').NoOfAccidents

vehicle_wise = vehicle_wise.sort_values('Total Accidents', ascending = False)
# Injuries and Fatalities as Percentages
vehicle_wise['Injury%'] = round((vehicle_wise['PERSONS INJURED']/vehicle_wise['Total Accidents'] * 100), 1)
vehicle_wise['Fatality%'] = round((vehicle_wise['PERSONS KILLED']/vehicle_wise['Total Accidents'] * 100), 3)

# Filtering vehicles involved in atleast 100 accidents
mask = vehicle_wise['Total Accidents'] > 100
vehicle_wise = vehicle_wise[mask]

vehicle_wise.head(3)




vehicle_accidents = vehicle_wise.sort_values('Total Accidents', ascending = False).head(10)
vehicle_accidents.head(3)




# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

# Creating the plot
vehicle_accidents.plot(x = 'VEHICLE 1 TYPE', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Vehicle types involved in the most number of accidents', size = 12)
ax.set_xlabel('Vehicle Type', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('vehicle_type_accidents.png', dpi=500)




vehicle_inj_rate = vehicle_wise.sort_values('Injury%', ascending = False).head(10)
vehicle_fat_rate = vehicle_wise.sort_values('Fatality%', ascending = False).head(10)




# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
color2 = cm.autumn(np.linspace(.2,.6, 10))

# Creating the plots
vehicle_inj_rate.plot(x = 'VEHICLE 1 TYPE', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

vehicle_fat_rate.plot(x = 'VEHICLE 1 TYPE', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Vehicle Types with the highest rate of injury', size = 12)
ax1.set_xlabel('Vehicle Types', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Vehicle Types with the highest rate of fatality', size = 12)
ax2.set_xlabel('Vehicle Types', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('vehicle_inj_fat_rate.png', dpi=500)




# Keeping only the date from the Datetime column
date_only = raw_data.copy() 
date_only['Date'] = date_only['CRASH_DATE_TIME'].dt.date

# Calculate the number of people killed, injured and total accidents for each contributing factor
date_wise = date_only.groupby(['Date'])['PERSONS KILLED', 'PERSONS INJURED'].agg('sum').reset_index()

date_wise['Total Accidents'] = date_only.groupby(['Date']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wise['Injury%'] = round((date_wise['PERSONS INJURED']/date_wise['Total Accidents'] * 100), 1)
date_wise['Fatality%'] = round((date_wise['PERSONS KILLED']/date_wise['Total Accidents'] * 100), 3)

date_wise = date_wise.sort_values('Total Accidents', ascending = False)




date_accidents = date_wise.sort_values('Total Accidents', ascending = False).head(10)

# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(12, 4))

# Defining color map
color = np.flip(cm.Oranges(np.linspace(.2,.6, 10)))

# Creating the plot
date_accidents.plot(x = 'Date', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Dates on which the most number of accidents occured', size = 12)
ax.set_xlabel('Date', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('date_accidents.png', dpi=500)




# Average number of accidents by date
date_wise['Total Accidents'].mean()




# Filtering dates with more than 604 accidents
mask = date_wise['Total Accidents'] > 604
vehicle_accidents_500 = date_wise[mask]
print("Average no. of people injured on dates when more than 536 accidents happened: " +       str(vehicle_accidents_500['PERSONS INJURED'].mean()))
print("Total people killed on dates when more than 536 accidents happened: " +       str(vehicle_accidents_500['PERSONS KILLED'].sum()))




# Dark Sky Secret Key
secret_key = '46d8abf841357ef2fe310170ad26ce87'

# NOTE: The key has been reset to avoid overuse. 
# Please create an account on DarkSky API to run the calls for yourself.




NYC_LAT = '40.730610'
NYC_LONG = '-73.935242'




vehicle_accidents_500_date = vehicle_accidents_500.copy()['Date'].head(10) # Limiting to first ten rows to speed up testing

frame = {'Date': vehicle_accidents_500_date} 
vehicle_accidents_500_date_df = pd.DataFrame(frame) 




casts = []

for date in vehicle_accidents_500_date_df['Date'].values.tolist():
    dt = str(date)
    date_time = dt + "T12:00:00"
    link = "https://api.darksky.net/forecast/{}/{},{},{}".format(secret_key, NYC_LAT, NYC_LONG, date_time)
    
    # Sending GET request and saving the response as a response object
    r = requests.get(url = link)
    
    # Unpacking data in JSON Format
    data = r.json() 
    to_cast = data['currently']['summary']
    
    casts.append(to_cast)

# Adding the result to the dataframe
vehicle_accidents_500_date_df['Summary'] = casts




# Joining the dataframes
vehicle_accidents_500_date_df = date_wise.set_index('Date').join(vehicle_accidents_500_date_df.set_index('Date'))




to_plot = vehicle_accidents_500_date_df.head(100)

# Calculate the number of people killed, injured and total accidents for each contributing factor
to_plot_grouped = to_plot.groupby(['Summary'])['PERSONS KILLED', 'PERSONS INJURED'].agg('sum').reset_index()

to_plot_grouped['Total Accidents'] = to_plot.groupby(['Summary']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
to_plot_grouped['Injury%'] = round((to_plot_grouped['PERSONS INJURED']/to_plot_grouped['Total Accidents'] * 100), 1)
to_plot_grouped['Fatality%'] = round((to_plot_grouped['PERSONS KILLED']/to_plot_grouped['Total Accidents'] * 100), 3)

to_plot_grouped = to_plot_grouped.sort_values('Total Accidents', ascending = False)




# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

# Creating the plot
to_plot_grouped.plot(x = 'Summary', y = 'Total Accidents', 
             kind = 'bar', color = color, 
             stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Weather Condition vs Number of Accidents', size = 12)
ax.set_xlabel('Weather Condition', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 90)

# Exporting the visualisation
fig.savefig('weather_summary_accidents.png', dpi=500)




weather_inj_rate = to_plot_grouped.sort_values('Injury%', ascending = False).head(10)
weather_inj = to_plot_grouped.sort_values('PERSONS INJURED', ascending = False).head(10)
weather_fat_rate = to_plot_grouped.sort_values('Fatality%', ascending = False).head(10)
weather_fat = to_plot_grouped.sort_values('PERSONS KILLED', ascending = False).head(10)




# Create figure and axes for Matplotlib
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
color2 = cm.PuRd(np.linspace(.2,.6, 10))

# Creating the plots
weather_inj_rate.plot(x = 'Summary', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

weather_inj.plot(x = 'Summary', y = 'PERSONS INJURED', kind = 'bar', 
                 color = color, stacked = True, ax = ax3)

weather_fat_rate.plot(x = 'Summary', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

weather_fat.plot(x = 'Summary', y = 'PERSONS KILLED', kind = 'bar', 
                 color = color2, stacked = True, ax = ax4)

# Customizing the Visulation
ax1.set_title('Weather Condition with the highest rate of injury', size = 12)
ax1.set_xlabel(' ', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Weather Condition with the highest rate of fatality', size = 12)
ax2.set_xlabel(' ', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

ax3.set_title('Weather Condition vs Injuries', size = 12)
ax3.set_xlabel('Weather Condition', size = 12)
ax3.set_ylabel('Number of Injured People', size = 12)
ax3.tick_params(labelrotation = 30)

ax4.set_title('Weather Condition vs Fatalities', size = 12)
ax4.set_xlabel('Weather Condition', size = 12)
ax4.set_ylabel('Number of Deaths', size = 12)
ax4.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('weather_inj_fat_rate.png', dpi=500)

