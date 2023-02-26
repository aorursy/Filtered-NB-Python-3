#!/usr/bin/env python
# coding: utf-8



# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")




# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """




# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)




# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")




print(accidents_by_day)




# Display a few rows to glance over the data.




accidents.head('accident_2015')




q_accidents_by_hour = '''
    SELECT
        EXTRACT(HOUR FROM timestamp_of_crash) hour,
        COUNT(consecutive_number) accidents
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
'''

accidents.estimate_query_size(q_accidents_by_hour)




accidents_by_hour_df = accidents.query_to_pandas(q_accidents_by_hour).sort_values('hour')




import seaborn as sns
sns.set()
accidents_by_hour_df.plot.bar(x='hour', title='Accidents by hour')




accidents.head('vehicle_2015')




q_hitruns_state = '''
    SELECT
        registration_state_name state_name,
        hit_and_run hit_and_run_happened,
        COUNT(*) hit_and_runs
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
    GROUP BY hit_and_run, registration_state_name
    HAVING hit_and_run='Yes'
    ORDER BY COUNT(*) DESC
    LIMIT 10;
'''

accidents.estimate_query_size(q_hitruns_state)




accidents.query_to_pandas(q_hitruns_state)

