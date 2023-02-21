#!/usr/bin/env python
# coding: utf-8



# Set your own project id here
PROJECT_ID = 'linear-rig-251417' # a string, like 'kaggle-bigquery-240818'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID




get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')




# We can train our model on data from one time period and test it on data collected after that period.




get_ipython().run_cell_magic('bigquery', 'dataframe_name', 'SELECT start_station_name, \n       TIMESTAMP_TRUNC(start_time, HOUR) as start_hour,\n       COUNT(bikeid) as num_rides\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time < "2018-01-01"\nGROUP BY start_station_name, start_hour')




dataframe_name.head()




get_ipython().run_cell_magic('bigquery', '', '\nCREATE OR REPLACE MODEL `model_dataset.bike_trips`\nOPTIONS(model_type=\'linear_reg\') AS\nSELECT COUNT(bikeid) as label, \n       start_station_name, \n       TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time < "2018-01-01"\nGROUP BY start_station_name, start_hour')




get_ipython().run_cell_magic('bigquery', '', '\nSELECT *\nFROM\nML.EVALUATE(MODEL `model_dataset.bike_trips`, (\nSELECT COUNT(bikeid) as label, \n       start_station_name, \n       TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time >= "2018-01-01"\nGROUP BY start_station_name, start_hour\n))')




## Maybe something has changed over time we're not accounting for.




get_ipython().run_cell_magic('bigquery', '', '\nSELECT AVG(ROUND(predicted_label)) as predicted_avg_riders, \n       AVG(label) as true_avg_riders\nFROM\nML.PREDICT(MODEL `model_dataset.bike_trips`, (\nSELECT COUNT(bikeid) as label,\n       start_station_name,\n       TIMESTAMP_TRUNC(start_time, HOUR) as start_hour\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time >= "2018-01-01" AND start_time < "2019-01-01"\n  AND start_station_name = "22nd & Pearl"\nGROUP BY start_station_name, start_hour\n))')




get_ipython().run_cell_magic('bigquery', '', '\nWITH daily_rides AS (\n    SELECT COUNT(bikeid) AS num_rides,\n           start_station_name,\n           EXTRACT(DAYOFYEAR from start_time) AS doy,\n           EXTRACT(YEAR from start_time) AS year\n    FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\n    GROUP BY start_station_name, doy, year\n    ORDER BY year\n), \nstation_averages AS (\n    SELECT avg(num_rides) AS avg_riders, start_station_name, year\n    FROM daily_rides\n    GROUP BY start_station_name, year)\n\nSELECT avg(avg_riders) AS daily_rides_per_station, year\nFROM station_averages\nGROUP BY year\nORDER BY year')




## Thought question answer here
## Yes

