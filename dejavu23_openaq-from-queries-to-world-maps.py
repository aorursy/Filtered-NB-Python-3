#!/usr/bin/env python
# coding: utf-8


















import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

# plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


from google.cloud import bigquery
from bq_helper import BigQueryHelper

bqh_openaq = BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

import pycountry

import folium


import os
print(os.listdir("../input"))




pollutants = ['o3','co','no2','so2','pm25_frm']









bqh_openaq.list_tables()




bqh_openaq.table_schema("global_air_quality")




bqh_openaq.head("global_air_quality", num_rows=10)









QUERY_aq_countries = """
                     SELECT DISTINCT country
                     FROM `bigquery-public-data.openaq.global_air_quality`
                     """




df_aq_countries = bqh_openaq.query_to_pandas(QUERY_aq_countries)
df_aq_countries.info()




df_aq_countries.head()









QUERY_aq_cities_us = """
                     SELECT DISTINCT city
                     FROM `bigquery-public-data.openaq.global_air_quality`
                     WHERE country = 'US'
                     """




df_aq_cities_us = bqh_openaq.query_to_pandas(QUERY_aq_cities_us)
df_aq_cities_us.info()




df_aq_cities_us.sample(5)









QUERY_openaq_all = """
                   SELECT *
                   FROM `bigquery-public-data.openaq.global_air_quality`                
                   """




bqh_openaq.estimate_query_size(QUERY_openaq_all)




df_openaq_all = bqh_openaq.query_to_pandas(QUERY_openaq_all)









df_openaq_all.shape




df_openaq_all.info()




df_openaq_all.sample(20)




df_openaq_all['timestamp'].min()




df_openaq_all['timestamp'].max()









df_openaq_all.set_index('timestamp', inplace=True)
df_openaq_all.sort_index(inplace=True)




df_openaq_all.head()




df_openaq_all.tail()




df_openaq_all.describe()




df_openaq_all[['value','pollutant']].groupby('pollutant').describe()




df_openaq_all = df_openaq_all[df_openaq_all['value'] > 0]  




df_openaq_all[['value','pollutant']].groupby('pollutant').describe()




df_openaq_all[['value','pollutant', 'unit']].groupby(['pollutant', 'unit']).describe()









pycountry.countries.get(alpha_2='DE')




c = pycountry.countries.get(alpha_2='DE')
c.name




def get_country_name(alph2):
    
    #temporary country code for Kosovo
    if alph2 == "XK":
        return "Kosovo"
    if alph2 == "CE":
        return "tbd"
    
    c = pycountry.countries.get(alpha_2=alph2)
    return c.name




df_openaq_all['country_name'] = df_openaq_all['country'].apply(get_country_name)




df_openaq_all['country_name'].value_counts().head(20)









df_openaq_all['pollutant'].value_counts()




df_openaq_all['unit'].value_counts()




cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_openaq_all.pollutant, df_openaq_all.unit, margins=True).style.background_gradient(cmap=cm)




cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_openaq_all.country_name, df_openaq_all.pollutant, margins=True).style.background_gradient(cmap=cm)




country_meas_count = pd.crosstab(df_openaq_all.country_name, df_openaq_all.pollutant)




country_meas_count.sort_values(by=['pm25'], ascending=False).head(10)




country_meas_count.sort_values(by=['pm10'], ascending=False).head(10)




country_meas_count.sort_values(by=['so2'], ascending=False).head(10)




country_meas_count.sort_values(by=['no2'], ascending=False).head(10)




country_meas_count.sort_values(by=['o3'], ascending=False).head(10)









df_2018 = df_openaq_all[df_openaq_all.index.year == 2018].copy()
df_2019 = df_openaq_all[df_openaq_all.index.year == 2019].copy()














def convert_pm10_to_AQI(val):
    
    C = round(val,1)
    
    I_breakpoints = [(0, 50) , (51, 100) , (101, 150), 
                     (151, 200) , (201, 300) , (301, 500) ]
    
    C_intervalls = [(0.0, 54.9) , (55.0, 154.9) , (155.0, 254.9), 
                    (255.0, 354.9) , (355.0, 424.9) , (425.0, 6000.0) ]
    
    for k in range(0,6):
    
        if C <= C_intervalls[k][1] and C >= C_intervalls[k][0] :
            I_low  = I_breakpoints[k][0]        
            I_high = I_breakpoints[k][1]
            C_low  = C_intervalls[k][0]        
            C_high = C_intervalls[k][1]        
           
   
    I = (I_high-I_low) / (C_high-C_low) * (C-C_low) + I_low
    I = round(I,1)
    
    return I
























def query_avg_1pollutant_1_year_all_countries_counts(pollutant, year) :
    
    s = "SELECT COUNT(value) as Count, country, AVG(value) as Concentration"
    f = " FROM `bigquery-public-data.openaq.global_air_quality` as globalAQ"
    w = " WHERE EXTRACT(YEAR FROM globalAQ.timestamp) = " + str(year) +         " AND unit = 'µg/m³'" + "AND pollutant = '" + pollutant + "'" + "AND value > 0"
    g = " GROUP BY country"
    q = s+f+w+g
    return q









q1 = query_avg_1pollutant_1_year_all_countries_counts("pm25", 2019) 
print(q1)




df_2019_pm25_mean = bqh_openaq.query_to_pandas(q1)
df_2019_pm25_mean.sort_values(by='Concentration', inplace=True)
df_2019_pm25_mean['country_name'] = df_2019_pm25_mean['country'].apply(get_country_name)
df_2019_pm25_mean = df_2019_pm25_mean[df_2019_pm25_mean['Count'] > 20]  




f,ax = plt.subplots(figsize=(14,6))
sns.barplot(df_2019_pm25_mean['country_name'],df_2019_pm25_mean['Concentration'])
plt.xticks(rotation=90)
plt.title('Mean concentration of PM2.5 in 2019');









df_2019_pm25 = df_2019[df_2019['pollutant'] == 'pm25'].copy()
df_2019_pm25['count'] = df_2019_pm25.groupby('country_name')['country_name'].transform('count')
df_2019_pm25.shape




df_2019_pm10 = df_2019[df_2019['pollutant'] == 'pm10'].copy()
df_2019_pm10['count'] = df_2019_pm10.groupby('country_name')['country_name'].transform('count')
df_2019_pm10.shape




df_2019_no2 = df_2019[df_2019['pollutant'] == 'no2'].copy()
df_2019_no2['count'] = df_2019_no2.groupby('country_name')['country_name'].transform('count')
df_2019_no2.shape




df_2019_o3 = df_2019[df_2019['pollutant'] == 'o3'].copy()
df_2019_o3['count'] = df_2019_o3.groupby('country_name')['country_name'].transform('count')
df_2019_o3.shape




def plotly_boxplots_sorted_by_yvals(df, catg_feature, sort_by_yvals, str_title, min_nr_meas=20):
    
    df = df[df['count'] > min_nr_meas]
    df_by_catg   = df.groupby([catg_feature])
    sortedlist_catg_str = df_by_catg[sort_by_yvals].median().sort_values().keys().tolist()
    
    sortedlist_catg_str = sortedlist_catg_str[-20:]
    
    data = []
    for i in sortedlist_catg_str :
        data.append(go.Box(y = df[df[catg_feature]==i][sort_by_yvals], name = i))

    layout = go.Layout(title = dict(text=str_title, xref="paper", x=0.5), 
                       yaxis = dict(title = sort_by_yvals + " (µg/m³)"),
                       showlegend=False)

    fig = dict(data = data, layout = layout)
    return fig




fig = plotly_boxplots_sorted_by_yvals(df_2019_pm25, 'country_name', 'value', 
                                      "PM 2.5 vs. country", min_nr_meas=20)
iplot(fig)




fig = plotly_boxplots_sorted_by_yvals(df_2019_pm10, 'country_name', 'value', 
                                      "PM 10 vs. country", min_nr_meas=20)
iplot(fig)




fig = plotly_boxplots_sorted_by_yvals(df_2019_no2, 'country_name', 'value', 
                                      "NO2 vs. country", min_nr_meas=20)
iplot(fig)









fig = plotly_boxplots_sorted_by_yvals(df_2019_o3, 'country_name', 'value', 
                                      "O3 vs. country", min_nr_meas=20)
iplot(fig)














def query_all_1pollutant_1country(pollutant, country) :
    
    s = "SELECT *"
    f = " FROM `bigquery-public-data.openaq.global_air_quality` as globalAQ"
    w = " WHERE unit = 'µg/m³'" + " AND pollutant = '" + pollutant + "'" +         " AND value > 0" + " AND country = '" + country + "'" +         " AND EXTRACT(YEAR FROM globalAQ.timestamp) > 2016"        
    q = s+f+w
    return q




q_m1 = query_all_1pollutant_1country("pm25", "CN")
q_m1




df_china_pm25 = bqh_openaq.query_to_pandas(q_m1)
df_china_pm25.sort_values(by=['timestamp'], inplace=True) 




df = df_china_pm25

mg_map = folium.Map(location=[35.0,115.0], tiles= "Stamen Terrain", zoom_start = 4.2)

for i in range(1,50):   
    lon = df.loc[df.index ==i]['longitude'].values[0]
    lat = df.loc[df.index ==i]['latitude'].values[0]
    folium.Marker([lat, lon]).add_to(mg_map)

mg_map



















def convert_pm25_to_AQI(val):
    
    C = round(val,1)
    
    I_breakpoints = [(0, 50) , (51, 100) , (101, 150), 
                     (151, 200) , (201, 300) , (301, 500) ]
    
    C_intervalls = [(0.0, 12.0) , (12.1, 35.4) , (35.5, 55.4), 
                    (55.5,150.4) , (150.5, 250.4) , (250.5, 2000.0) ]
    
    for k in range(0,6):
    
        if C <= C_intervalls[k][1] and C >= C_intervalls[k][0] :
            I_low  = I_breakpoints[k][0]        
            I_high = I_breakpoints[k][1]
            C_low  = C_intervalls[k][0]        
            C_high = C_intervalls[k][1]        
           
   
    I = (I_high-I_low) / (C_high-C_low) * (C-C_low) + I_low
    I = round(I,1)
    
    return I




convert_pm25_to_AQI(12.04)




df_china_pm25['AQI'] = df_china_pm25['value'].apply(convert_pm25_to_AQI)




AQI_categories = ['Good', 'Moderate', 'Unhealthy for sensitive groups', 
                  'Unhealthy', 'Very Unhealthy', 'Hazardous']

AQI_breakpoints = [50, 100, 150, 200, 300]




def apply_AQI_category(df) :
    
    df['AQI_category'] = "Good"
    df.loc[(df['AQI'] > 50) , 'AQI_category'] = 'Moderate'    
    df.loc[(df['AQI'] > 100), 'AQI_category'] = 'Unhealthy for sensitive groups'
    df.loc[(df['AQI'] > 150), 'AQI_category'] = 'Unhealthy'    
    df.loc[(df['AQI'] > 200), 'AQI_category'] = 'Very Unhealthy'
    df.loc[(df['AQI'] > 300), 'AQI_category'] = 'Hazardous'
    
    return df




df_china_pm25 = apply_AQI_category(df_china_pm25)




df_china_pm25[['value', 'AQI', 'AQI_category']].sample(10)









EPA_AQI_PM25_step_colormap =     folium.StepColormap( ['green','yellow','orange','red','purple','brown'], 
                         vmin=0., vmax=300. ,
                         index=[0, 12, 35.4, 55.4, 150.4, 250.4] ,
                         caption='PM 2.5'
                        )

EPA_AQI_PM25_step_colormap




EPA_AQI_step_colormap =     folium.StepColormap( ['green','yellow','orange','red','purple','brown'], 
                         vmin=0., vmax=400. ,
                         index=[0, 51, 101, 151, 201, 301],
                         caption='AQI'
                        )
EPA_AQI_step_colormap









m = folium.Map(location=[36.0,108.0], tiles= "Stamen Toner", zoom_start = 3.7)

for index, row in df.iterrows():
    folium.CircleMarker( [row['latitude'], row['longitude']] , radius=3, 
                         color=EPA_AQI_PM25_step_colormap(row['value']), fill=True, fill_opacity=1.0,            
                         fill_color=EPA_AQI_PM25_step_colormap(row['value']), popup=row['city'] ).add_to(m)
EPA_AQI_PM25_step_colormap.caption = 'PM 2.5'
EPA_AQI_PM25_step_colormap.add_to(m)    
m









def folium_AQI_map(df, center_lat, center_lon, zoom) :
    
    m = folium.Map(location=[center_lat, center_lon],
                   tiles= "cartodbpositron",
                   zoom_start = zoom)

    for index, row in df.iterrows():
        folium.CircleMarker( [row['latitude'], row['longitude']] , radius=3, 
                             color=EPA_AQI_step_colormap(row['AQI']), 
                             fill=True, fill_opacity=1.0,            
                             fill_color=EPA_AQI_step_colormap(row['AQI']), 
                             popup=row['city'] ).add_to(m)

    EPA_AQI_step_colormap.add_to(m)

    
    folium.TileLayer(tiles='Stamen Toner',name="Stamen Toner").add_to(m)
    folium.TileLayer(tiles='Stamen Terrain',name="Stamen Terrain").add_to(m)
    folium.LayerControl().add_to(m)    
    
    return m
    




map_china_pm25 = folium_AQI_map(df_china_pm25, 36.0, 108.0, 3.9)
map_china_pm25




df_china_pm25.sort_values(by="AQI", ascending=False).head(10)




q_m2 = query_all_1pollutant_1country("pm25", "US")
q_m2




df_usa_pm25 = bqh_openaq.query_to_pandas(q_m2)
df_usa_pm25.sort_values(by=['timestamp'], inplace=True) 
df_usa_pm25['AQI'] = df_usa_pm25['value'].apply(convert_pm25_to_AQI)
df_usa_pm25 = apply_AQI_category(df_usa_pm25)




map_usa_pm25 = folium_AQI_map(df_usa_pm25, 38.0,-101.0, 4.0)
map_usa_pm25




df_usa_pm25.sort_values(by="AQI", ascending=False).head(10)









q_m3 = query_all_1pollutant_1country("pm25", "IN")
q_m3




df_india_pm25 = bqh_openaq.query_to_pandas(q_m3)
df_india_pm25.sort_values(by=['timestamp'], inplace=True) 
df_india_pm25['AQI'] = df_india_pm25['value'].apply(convert_pm25_to_AQI)
df_india_pm25 = apply_AQI_category(df_india_pm25)




map_india_pm25 = folium_AQI_map(df_india_pm25, 21.0,77.0, 4.0)
map_india_pm25




df_india_pm25.sort_values(by="AQI", ascending=False).head(10)









def query_all_1pollutant_geobox(pollutant, arr) :
    
    lat_min, lat_max = arr[0], arr[1]
    lon_min, lon_max = arr[2], arr[3]   
    
    print("lat_min, lat_max : ", lat_min, lat_max)
    print("lon_min, lon_max : ", lon_min, lon_max)
    
    s = "SELECT *"
    f = " FROM `bigquery-public-data.openaq.global_air_quality` as globalAQ"
    w = " WHERE unit = 'µg/m³'" + " AND value > 0" +         " AND pollutant = '" + pollutant + "'" +         " AND latitude > " + str(lat_min) +         " AND latitude < " + str(lat_max) +         " AND longitude > " + str(lon_min) +         " AND longitude < " + str(lon_max) +         " AND EXTRACT(YEAR FROM globalAQ.timestamp) > 2016"        
    q = s+f+w
    return q




q_m4 = query_all_1pollutant_geobox("pm25", [10.0, 75.0, -20.0, 40.0])
q_m4




df_europe_pm25 = bqh_openaq.query_to_pandas(q_m4)
df_europe_pm25.sort_values(by=['timestamp'], inplace=True) 
df_europe_pm25['AQI'] = df_europe_pm25['value'].apply(convert_pm25_to_AQI)
df_europe_pm25 = apply_AQI_category(df_europe_pm25)




map_europe_pm25 = folium_AQI_map(df_europe_pm25, 56.0,10.0, 3.4)
map_europe_pm25









df_europe_pm25.sort_values(by="AQI", ascending=False).head(10)




df_openaq_all.tail()




latest_year  = df_openaq_all.iloc[-1:].index.year.values[0]
latest_month = df_openaq_all.iloc[-1:].index.month.values[0]
latest_day   = df_openaq_all.iloc[-1:].index.day.values[0]




print(latest_year, latest_month, latest_day)




import datetime
now = datetime.datetime.now()
print(now.year)




if latest_year > 2019:
    latest_year   = now.year
    latest_month  = now.month
    









def query_all_1pollutant_prevmonth(pollutant, y, m) :
    
    s = "SELECT *"
    f = " FROM `bigquery-public-data.openaq.global_air_quality` as globalAQ"
    w = " WHERE unit = 'µg/m³'" + " AND pollutant = '" + pollutant + "'" +         " AND value > 0"  +         " AND EXTRACT(YEAR FROM globalAQ.timestamp) = " + str(y)  +         " AND EXTRACT(MONTH FROM globalAQ.timestamp) > " + str(m-2)   
    
    q = s+f+w
    return q




q_w1 = query_all_1pollutant_prevmonth("pm25", latest_year, latest_month)
q_w1




df_world_pm25_prevmonth = bqh_openaq.query_to_pandas(q_w1)
df_world_pm25_prevmonth.sort_values(by=['timestamp'], inplace=True) 
df_world_pm25_prevmonth['AQI'] = df_world_pm25_prevmonth['value'].apply(convert_pm25_to_AQI)




df_world_pm25_prevmonth = df_world_pm25_prevmonth.sample(1500)




map_world_pm25 = folium_AQI_map(df_world_pm25_prevmonth, 0.0,0.0, 1.5)
map_world_pm25



















q_pm10_cn = query_all_1pollutant_1country("pm10", "CN")
q_pm10_cn




df_china_pm10 = bqh_openaq.query_to_pandas(q_pm10_cn)
df_china_pm10.sort_values(by=['timestamp'], inplace=True) 
df_china_pm10['AQI'] = df_china_pm10['value'].apply(convert_pm10_to_AQI)
df_china_pm10 = apply_AQI_category(df_china_pm10)




map_china_pm10 = folium_AQI_map(df_china_pm10, 36.0, 108.0, 3.9)
map_china_pm10









df_china_pm10.sort_values(by="AQI", ascending=False).head(10)









q_pm10_usa = query_all_1pollutant_1country("pm10", "US")
q_pm10_usa




df_usa_pm10 = bqh_openaq.query_to_pandas(q_pm10_usa)
df_usa_pm10.sort_values(by=['timestamp'], inplace=True) 
df_usa_pm10['AQI'] = df_usa_pm10['value'].apply(convert_pm10_to_AQI)
df_usa_pm10 = apply_AQI_category(df_usa_pm10)




map_usa_pm10 = folium_AQI_map(df_usa_pm10, 38.0,-101.0, 4.0)
map_usa_pm10




df_usa_pm10.sort_values(by="AQI", ascending=False).head(10)









q_pm10_in = query_all_1pollutant_1country("pm10", "IN")
q_pm10_in




df_india_pm10 = bqh_openaq.query_to_pandas(q_pm10_in)
df_india_pm10.sort_values(by=['timestamp'], inplace=True) 
df_india_pm10['AQI'] = df_india_pm10['value'].apply(convert_pm10_to_AQI)
df_india_pm10 = apply_AQI_category(df_india_pm10)




map_india_pm10 = folium_AQI_map(df_india_pm10, 21.0,77.0, 4.0)
map_india_pm10




df_india_pm10.sort_values(by="AQI", ascending=False).head(10)









q_pm10_eu = query_all_1pollutant_geobox("pm10", [10.0, 75.0, -20.0, 40.0])
q_pm10_eu




df_europe_pm10 = bqh_openaq.query_to_pandas(q_pm10_eu)
df_europe_pm10.sort_values(by=['timestamp'], inplace=True) 
df_europe_pm10['AQI'] = df_europe_pm10['value'].apply(convert_pm10_to_AQI)
df_europe_pm10 = apply_AQI_category(df_europe_pm10)




df_europe_pm10.sort_values(by="AQI", ascending=False).head(10)
















