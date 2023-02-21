#!/usr/bin/env python
# coding: utf-8



# Set your own project id here
PROJECT_ID = 'bqml-exercise-rental-bikes' # a string, like 'kaggle-bigquery-240818'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID




get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')




# You can write your notes here




table=client.get_table('bigquery-public-data.austin_bikeshare.bikeshare_trips')
#df=client.list_rows(table).to_dataframe()
client.list_rows(table,max_results=5).to_dataframe()




#client.list_rows(table).to_dataframe()['start_station_id']==0




get_ipython().run_cell_magic('bigquery', 'dataset_name', "SELECT start_station_name,\n       TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,\n       COUNT(bikeid) AS num_rides\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time < '2018-01-01'\nGROUP BY start_station_name, start_hour")




dataset_name.head()




"""CREATE OR REPLACE MODEL `model_dataset.bike_trips`
OPTIONS(model_type='linear_reg',input_label_cols=['num_rides']) AS 
SELECT start_station_name,
       TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,
       COUNT(trip_id) AS num_rides
FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
WHERE EXTRACT (YEAR FROM start_time) < 2018
GROUP BY start_station_name, start_hour"""




get_ipython().run_cell_magic('bigquery', '', "CREATE OR REPLACE MODEL `model_dataset.bike_trips`\nOPTIONS(model_type='linear_reg') AS \nSELECT start_station_name,\n       TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,\n       COUNT(bikeid) AS label\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time < '2018-01-01'\nGROUP BY start_station_name, start_hour")




get_ipython().run_cell_magic('bigquery', '', 'SELECT\n  *\nFROM\n  ML.TRAINING_INFO(MODEL `model_dataset.bike_trips`)\nORDER BY iteration ')




get_ipython().run_cell_magic('bigquery', '', "SELECT\n  *\nFROM ML.EVALUATE(MODEL `model_dataset.bike_trips`,(\n  SELECT start_station_name,\n       TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,\n       COUNT(bikeid) AS label\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time >= '2018-01-01'\nGROUP BY start_station_name, start_hour))")




get_ipython().run_cell_magic('bigquery', 'twentysecond', "SELECT\n  *\nFROM\n  ML.PREDICT(MODEL `model_dataset.bike_trips`,(\n  SELECT start_station_name,\n       TIMESTAMP_TRUNC(start_time, HOUR) AS start_hour,\n       COUNT(bikeid) AS label\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nWHERE start_time >= '2018-01-01'\nGROUP BY start_station_name, start_hour))\nWHERE start_station_name='22nd & Pearl'")




twentysecond_pred=twentysecond['predicted_label'].mean()
twentysecond_act=twentysecond['label'].mean()
print('Mean value of predicted riders:\t\t',twentysecond_pred)
print('Mean value of actual riders:\t\t ',twentysecond_act)




get_ipython().run_cell_magic('bigquery', 'trend', 'SELECT start_station_name,\n       EXTRACT(YEAR FROM start_time) AS year,\n       COUNT(bikeid) AS num_rides_per_year\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`\nGROUP BY start_station_name, year\nORDER BY year')




trend.groupby('year').mean()




trend['avg_daily_rides']=trend.num_rides_per_year/365

