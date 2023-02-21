#!/usr/bin/env python
# coding: utf-8



# Set your own project id here
PROJECT_ID = 'bigquery-bikes' # a string, like 'kaggle-bigquery-240818'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID




get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')




# We would like to split the dataset into a training dataset and a test dataset.
# Ideally, the ratio between training and test data should be in the order of 80:20, or 2/3 : 1/3.
# I would sort data by ascending datetime and use the older values in the training dataset and the most recent values in the test set.
# Depending on the data available, I'd try to gather at least one full year of data in the training dataset (to try to capture seasonality).




# create a reference to our table
table = client.get_table("bigquery-public-data.austin_bikeshare.bikeshare_trips")

# look at five rows from our dataset
client.list_rows(table, max_results=5).to_dataframe()




get_ipython().run_cell_magic('bigquery', 'training_dataset', 'SELECT\n  IFNULL(start_station_name, "") as start_station_name,\n  TIMESTAMP_TRUNC(start_time, HOUR) as start_hour,\n  COUNT(1) as num_rides\nFROM\n  `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE\n  start_time < \'2018-01-01\'\nGROUP BY start_station_name, start_hour\nORDER BY start_station_name, start_hour   ')




training_dataset.head()




training_dataset.info()




training_dataset['start_station_name'].nunique()




training_dataset['start_station_name'].value_counts()




# There are names of stations that look like types of use more than real station names: eg. "Repair Shop", "Re-branding", "Customer Service",
#"Marketing Event", "Stolen".
# also, there are mobile stations at different locations. Maybe we could group these data altogether and discard them from the model? 




training_dataset['start_hour'].unique()




get_ipython().run_cell_magic('bigquery', '', 'CREATE OR REPLACE MODEL`model_dataset.bike_trips`\nOPTIONS(model_type=\'linear_reg\') AS \nSELECT\n  COUNT(1) as label,\n  IFNULL(start_station_name, "") as start_station_name,\n  TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\nFROM\n  `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE\n  start_time < \'2018-01-01\'\nGROUP BY start_station_name, start_hour \nORDER BY start_station_name, start_hour ')




get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM ML.EVALUATE(MODEL `model_dataset.bike_trips`, (\n  SELECT  \n    COUNT(1) as label,\n    IFNULL(start_station_name, "") as start_station_name,\n    TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\n  FROM\n    `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n  WHERE\n      start_time > \'2018-01-01\'\nGROUP BY start_station_name, start_hour\nORDER BY start_station_name, start_hour ))')




## The poor performance might come from:

## * Issues in the training and test datasets, such as too little data, missing data, or erroneous data. We may have to discard stations with too
## little data or outliers.

##* A change in patterns of usage between the training period and the test period, that could result from changes in the context (eg: new tariff scheme,
## introduction of biking incentives, opening of new stations...). For example, a steep increase in usage, as more and more people adopt the bike sharing 
## system and new stations are introduced.

## * A change in patterns of usage between the training period and the test period, since we're dealing here with a growing network (of bike sharing):
## Starting with a dozen of stations in 2013, the network has more than 80 stations in 2018.
## Therefore, as more and more stations open, the bike sharing network might get denser, and people adopt new stations that are closer. We might observe
## a decline in daily average in older stations that are close to newer stations.

## * Linear model might not be appropriate.

## * Events occurring at different dates each year: Are they any events that might significantly impact shared bike usage? If so, such events
## might not be identified in our variables, since dates differ each year.

## * The number of variables might be too low, and we might need more features in our model to explain patterns.




get_ipython().run_cell_magic('bigquery', '', '\nSELECT AVG(ROUND(predicted_label)) as predicted_avg_riders, \n       AVG(label) as true_avg_riders\nFROM\nML.PREDICT(MODEL `model_dataset.bike_trips`, (\nSELECT COUNT(1) as label,\n       start_station_name,\n       TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time >= "2018-01-01" AND start_time < "2019-01-01"\n  AND start_station_name = "22nd & Pearl"\nGROUP BY start_station_name, start_hour\n))')




get_ipython().run_cell_magic('bigquery', 'avg_daily_rides_per_date', 'SELECT\n  TIMESTAMP_TRUNC(start_time, DAY) as start_date,\n  COUNT(1) / COUNT(DISTINCT(IFNULL(start_station_name, ""))) as avg_daily_rides\nFROM\n  `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nGROUP BY start_date\nORDER BY start_date')




avg_daily_rides_per_date.set_index('start_date', inplace = True)




import matplotlib.pyplot as plt
fig, axes = plt.subplots(figsize=(12,3))
axes.plot(avg_daily_rides_per_date.index, avg_daily_rides_per_date['avg_daily_rides'])
axes.set_ylabel('Average daily rides per station')
axes.set_title('Average daily rides per station over the study period');




## The pattern in 2018 looks very different from the pattern during the other years. There is a sustained high level of daily rides during a few months from
## mid-February to mid-May (and to a lesser extend in September-October) that was not observed during previous years.




## Please see my post in the Kaggle Learn Forums for ideas of improvements: https://www.kaggle.com/learn-forum/103648.
## Also below is the code to plot the daily riders for each year of the study period for the following datasets:
## 1/ dataset:  stations existing prior to 2018, that are not temporary (i.e. not 'Mobile Station...', 'MapJam...'), and that have more than 100 riders
## per year
## 2/ dataset:  stations new in 2018, that are not temporary (i.e. not 'Mobile Station...', 'MapJam...'), and that have more than 100 riders per year




# Extract the first dataset (stations existing prior to 2018...)
SELECT
  EXTRACT(YEAR from start_time) as year,
  EXTRACT(MONTH from start_time) as month,
  EXTRACT(DAY from start_time) as day,
  EXTRACT(DAYOFYEAR FROM start_time) as day_of_year,
  COUNT(1) as num_rides
FROM
  `bigquery-public-data.austin_bikeshare.bikeshare_trips` 
WHERE (start_station_name NOT IN ('10th & Red River',
                                 '11th & Salina',
                                 '11th & Salina ',
                                 '21st & Speedway @PCL',
                                 '21st & University',
                                 '22nd & Pearl',
                                 '23rd & Rio Grande',
                                 '23rd & San Jacinto @ DKR Stadium',
                                 '5th & Campbell', '6th & Chalmers',
                                 '6th & Chalmers ', '8th & Lavaca',
                                 'Dean Keeton & Speedway',
                                 'Dean Keeton & Speedway ',
                                 'Dean Keeton & Whitis',
                                 "Eeyore's 2017", "Eeyore's 2018",
                                 'Hollow Creek & Barton Hills',
                                 'Lake Austin & Enfield', 'Lake Austin Blvd @ Deep Eddy',
                                 'Lakeshore & Pleasant Valley', 'Lakeshore @ Austin Hostel',
                                 'Nash Hernandez @ RBJ South', 'Nueces & 26th', 
                                 'Red River/Cesar Chavez @ The Fairmont',
                                 'Repair Shop', 'Rio Grande & 28th',
                                 'Rosewood & Angelina', 'Rosewood & Chicon',
                                 'South Congress @ Bouldin Creek'))
  AND (start_station_name NOT IN ('6th & Chalmers ', '6th & Congress', 'Customer Service',
       'East 7th & Pleasant Valley', "Eeyore's 2017", "Eeyore's 2018",
       'Main Office', 'MapJam at French Legation',
       'MapJam at Pan Am Park', 'MapJam at Scoot Inn', 'Marketing Event',
       'Re-branding', 'Repair Shop', 'Shop', 'Stolen'))
  AND (start_station_name NOT IN ('MapJam at Hops & Grain Brewery', 'Mobile Station',
'Mobile Station @ Bike Fest',
       'Mobile Station @ Boardwalk Opening Ceremony',
       'Mobile Station @ Unplugged'))
GROUP BY year, month, day, day_of_year
ORDER BY year, day_of_year 




# Store results in dataframes and add dates (there should be a simpler way to code it...)
from datetime import datetime
import numpy as np

daily_riders = dict()
for y in range(2013,2020):
    # extract 1 year
    daily_riders[y] = overall_daily_riders_re_sampled[overall_daily_riders_re_sampled['year']==y]
    # retrieve dates
    daily_riders[y]['date'] = '1901-01-01'
    for i in range(len(daily_riders[y])):
        daily_riders[y]['date'].iloc[i] =datetime(daily_riders[y]['year'].iloc[i], daily_riders[y]['month'].iloc[i], daily_riders[y]['day'].iloc[i])




# plot results for 2018
y = 2018
import matplotlib.pyplot as plt
plt.plot(daily_riders[y]['date'],daily_riders[y]['num_rides'])




# plot for all the year using plotly
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly import subplots

path = "" # insert path here
export_filename =  path+"daily_riders_over_the_years.html"
overall_title = 'Daily riders from selected start stations (existing prior to 2018, not temporary, with more than 100 riders per year)'

fig = subplots.make_subplots(rows=7, cols=1)

for i in range(7):
    y = 2013 + i
    trace = go.Bar(
        x=daily_riders[y]['date'],
        y=daily_riders[y]['num_rides'],
        #marker = dict(color =daily_riders[y]['colour']),
        name = y
    )
    
    fig.append_trace(trace, i+1, 1)

y_max = max([daily_riders[y]['num_rides'].max() for y in range(2013,2020)])
fig['layout'].update(title=overall_title)
fig['layout']['yaxis1'].update(range=[0, y_max])

for i in range(7):
    y = 2013 + i
    fig['layout']['xaxis'+str(i+1)].update(range=[str(y)+'-01-01', str(y)+'-12-31'])
    fig['layout']['yaxis'+str(i+1)].update(range=[0, y_max], title = str(y))

pyo.plot(fig)
#pyo.plot(fig,filename = export_filename)




# Extract the second dataset (stations new in 2018...)
# same but "IN" instead of "NOT In" in the first WHERE condition
"""
WHERE (start_station_name IN ('10th & Red River',
                                 '11th & Salina',
                                 '11th & Salina ',
                                 '21st & Speedway @PCL',
                                 '21st & University',
                                 '22nd & Pearl',
                                 '23rd & Rio Grande',
                                 '23rd & San Jacinto @ DKR Stadium',
                                 '5th & Campbell', '6th & Chalmers',
                                 '6th & Chalmers ', '8th & Lavaca',
                                 'Dean Keeton & Speedway',
                                 'Dean Keeton & Speedway ',
                                 'Dean Keeton & Whitis',
                                 "Eeyore's 2017", "Eeyore's 2018",
                                 'Hollow Creek & Barton Hills',
                                 'Lake Austin & Enfield', 'Lake Austin Blvd @ Deep Eddy',
                                 'Lakeshore & Pleasant Valley', 'Lakeshore @ Austin Hostel',
                                 'Nash Hernandez @ RBJ South', 'Nueces & 26th', 
                                 'Red River/Cesar Chavez @ The Fairmont',
                                 'Repair Shop', 'Rio Grande & 28th',
                                 'Rosewood & Angelina', 'Rosewood & Chicon',
                                 'South Congress @ Bouldin Creek'))
"""




# to plot, follow the same steps as above. Just change the formula for y_max:
# y_max = max([daily_riders[y]['num_rides'].max() for y in range(2018,2020)])






