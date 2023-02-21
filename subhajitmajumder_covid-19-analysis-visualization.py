#!/usr/bin/env python
# coding: utf-8



#  pip install calmap




# pip install folium




# pip install plotly==4.6.0




# pip install "notebook>=5.3" "ipywidgets==7.5"




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import random
from datetime import timedelta

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas.plotting import register_matplotlib_converters
from plotly import tools

import json, requests
# import calmap
import folium

#offline plotting

from plotly.offline import plot, iplot, init_notebook_mode
from IPython.core.display import HTML
from IPython.display import HTML


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




df_full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv',
                parse_dates=['Date'])
# df_full_table.head(5)




#USA provinces data

df_usa_table = pd.read_csv('../input/corona-virus-report/usa_county_wise.csv')
# df_usa_table




#China Provinces Data

# df_china_provinces = pd.read_csv('https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/china_province_wise.csv')
# df_china_provinces




#Github loading
# df_time_series_table = pd.read_csv('https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/old/time_series_covid19_confirmed_global.csv')
# df_time_series_table




# india_data1 = requests.get('https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States').json()
# df_india1 = pd.io.json.json_normalize(india_data['data']['statewise'])
# df_india1 = df_india.set_index('states')
# india_data1




# df_indian_state = pd.read_csv('https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States')









#Active Case = confirmed - deaths - recovered
df_full_table['Active'] =df_full_table['Confirmed'] - df_full_table['Deaths'] - df_full_table['Recovered']

#Replacing Mainland China with China
df_full_table['Country/Region'] = df_full_table['Country/Region'].replace('Mainland China', 'China')

#Filling missing values
df_full_table[['Province/State']] = df_full_table[['Province/State']].fillna('')
df_full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']] = df_full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0) 

#Calculate Mortality Rate
df_full_table["Mortality Rate(Per 100)"] = np.round(100*df_full_table["Deaths"]/df_full_table["Confirmed"],2)
df_full_table

df_full_table.sample(5)




df_usa_table['Mortality Rate(per 100)'] = np.round(100*df_usa_table['Deaths']/df_usa_table['Confirmed'], 2)
# df_usa_table




#Grouped by day, country
df_full_group = df_full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

#new cases
temporary_new_cases = df_full_group.groupby(['Country/Region', 'Date'])['Confirmed', 'Deaths', 'Recovered']
temporary_new_cases = temporary_new_cases.sum().diff().reset_index()

mask = temporary_new_cases['Country/Region'] != temporary_new_cases['Country/Region'].shift(1)

temporary_new_cases.loc[mask, 'Confirmed'] = np.nan
temporary_new_cases.loc[mask, 'Deaths'] = np.nan
temporary_new_cases.loc[mask, 'Recovered'] = np.nan

#Renaming Columns
temporary_new_cases.columns = ['Country/Region', 'Date', 'New cases',
                              'New Deaths', 'New Recovered']

#Merging new values
df_full_group = pd.merge(df_full_group, temporary_new_cases, on=['Country/Region', 'Date'])

#Filling na with 0
df_full_group = df_full_group.fillna(0)


# df_full_group.head()




#Calculate Mortlity Rate
df_full_group["Mortality Rate(Per 100)"] = np.round(100*df_full_group["Deaths"]/df_full_group["Confirmed"],2)
# df_full_group




#Calculate Mortlity Rate China provinces
# df_china_provinces["Mortality Rate(Per 100)"] = np.round(100*df_china_provinces["Deaths"]/df_china_provinces["Confirmed"],2)
# df_china_provinces




#Country wise

#Latest values
country_wise = df_full_group[df_full_group['Date'] == max(df_full_group['Date'])].reset_index(drop=True).drop('Date', axis=1)
country_wise

#group by country
country_wise = country_wise.groupby('Country/Region')['Confirmed',
                                                     'Deaths',
                                                     'Recovered',
                                                     'Active',
                                                     'New cases',
                                                     'New Deaths',
                                                     'New Recovered',
                                                     'Mortality Rate(Per 100)'].sum().reset_index()

country_wise




temp = df_full_group.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars='Date', value_vars=['Recovered', 'Deaths', 'Active'],
                var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x='Date', y='Count', color='Case', height=800,width = 800,
             title='Cases over time')
fig.update_layout(xaxis_rangeslider_visible=True,
                 annotations = [dict(x = '2020-03-11', y = 468, 
                                    xref = 'x', yref = 'y', 
                                    text = 'Decleared Pandemic by WHO (11th March)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 12,
                                     color = '#000000'),
                                    ax = -100, ay=-150)])
fig.show()




temp1 = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)
temp1




fig = px.pie(temp1, values = 'Count', names = 'Case',
            color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()




# df_full_table.head()




country_wise = df_full_group[df_full_group['Date'] == max(df_full_group['Date'])].reset_index(drop=True)




country_wise.sort_values('Confirmed', ascending = False).style.background_gradient(cmap='Wistia', subset=['Confirmed']).background_gradient(cmap='Reds', subset=['Deaths']).background_gradient(cmap='Greens', subset=['Recovered']).background_gradient(cmap='Oranges', subset=['Active']).background_gradient(cmap='Purples', subset=['New cases']).background_gradient(cmap='OrRd', subset=['New Deaths'])




HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1852777" data-url="https://flo.uri.sh/visualisation/1852777/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')




temp_date = df_full_table[df_full_table['Date'] == max(df_full_table['Date'])]
# temp_date.head(1)
m = folium.Map(location=[10, 0], tiles='cartodbpositron',
              min_zoom=2, max_zoom=6, zoom_start=2.3)
for i in range(0, len(temp_date)):
    folium.Circle(
    location=[temp_date.iloc[i]['Lat'], temp_date.iloc[i]['Long']],
    color='crimson', fill='crimson',
    tooltip = "<h5 style='text-align:center;font-weight: bold'>Country :"+str(temp_date.iloc[i]['Country/Region'])+"</h5>"+    
    '<li><bold>Confirmed : '+str(temp_date.iloc[i]['Confirmed'])+
    '<li><bold>Fatalities : '+str(temp_date.iloc[i]['Deaths'])+
    '<li><bold>Active Cases : '+str(temp_date.iloc[i]['Active'])+
    '<li><bold>Recovered : '+str(temp_date.iloc[i]['Recovered']),
    radius = int(temp_date.iloc[i]['Confirmed'])**1.0).add_to(m)
m




# df_full_table.loc[df_full_table['Country/Region'] == 'US']




temp_date = df_full_table[df_full_table['Date'] == max(df_full_table['Date'])]
fig = px.scatter_mapbox(temp_date, lat="Lat", lon="Long", hover_name="Country/Region", hover_data=["Country/Region","Province/State","Confirmed"],
                        color_discrete_sequence=["fuchsia"], zoom=1, height=500,title='Confirmed count of each country' )
fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ])
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()




fig = px.bar(country_wise.sort_values('Confirmed', ascending=False)[:10][::-1],
            x = 'Country/Region', y = 'Confirmed',
            title = 'Top 10 countries (Confirmed Cases)',
            height = 500,width = 600, orientation = 'v', text = 'Confirmed',
            color_discrete_sequence = ['#FFC125'])
            
fig.show()




fig = px.bar(country_wise.sort_values('Deaths', ascending=False)[:10][::-1],
            x = 'Country/Region', y = 'Deaths',
            title = 'Top 10 countries (Death Cases)',
            height = 500,width = 600, orientation = 'v', text = 'Deaths',
            color_discrete_sequence = ['#ff3030'])
            
fig.show()




fig = px.bar(country_wise.sort_values('Recovered', ascending=False)[:10][::-1],
            x = 'Country/Region', y = 'Recovered',
            title = 'Top 10 countries (Recovered Cases)',
            height = 500,width = 600, orientation = 'v', text = 'Recovered',
            color_discrete_sequence = ['#2b8622'])
            
fig.show()




fig = px.bar(country_wise.sort_values('Active', ascending=False)[:10][::-1],
            x = 'Country/Region', y = 'Active',
            title = 'Top 10 countries (Active Cases)',
            height = 500,width = 600, orientation = 'v', text = 'Active',
            color_discrete_sequence = ['#004c4c'])
            
fig.show()




fig = px.bar(country_wise.sort_values('Recovered', ascending=False)[:10][::-1],
            x = 'Country/Region', y = 'Mortality Rate(Per 100)',
            title = 'Top 10 countries (Mortality Rates)',
            height = 500,width = 600, orientation = 'v', text = 'Mortality Rate(Per 100)',
            color_discrete_sequence = ['#800000'])
            
fig.show()




# country_wise




#China Data
df_china = df_full_table[df_full_table['Country/Region'] == 'China']
df_china

df_grouped_china = df_china.groupby('Date')['Country/Region','Date', 'Confirmed', 'Deaths'].sum().reset_index()
# df_grouped_china




fig = px.line(df_grouped_china, x = 'Date', y = 'Confirmed',
             title = 'Confirmed cases of China',
             color_discrete_sequence = ['#C70039'],
             height=500, width = 900)
fig.show()




china_provinces = df_china[df_china['Date'] == max(df_china['Date'])].reset_index(drop=True).drop(['Lat','Long','Country/Region'], axis = 1)
# china_provinces

# df_china[df_china['Date'] == '2020-04-18']




china_provinces.sort_values('Confirmed', ascending = False).style.background_gradient(cmap='Wistia', subset=['Confirmed']).background_gradient(cmap='Reds', subset=['Deaths']).background_gradient(cmap='Greens', subset=['Recovered']).background_gradient(cmap='Oranges', subset=['Mortality Rate(Per 100)']).background_gradient(cmap='ocean', subset=['Active'])




temp_date = df_china[df_china['Date'] == max(df_china['Date'])]
temp_date
m = folium.Map(location=[31, 117], tiles='cartodbpositron',
              min_zoom=3, max_zoom=5, zoom_start=4)
for i in range(0, len(temp_date)):
    folium.Circle(
    location=[temp_date.iloc[i]['Lat'], temp_date.iloc[i]['Long']],
    color='#FF5733 ',fill='crimson',
    tooltip = '<li><bold>Country : '+str(temp_date.iloc[i]['Country/Region'])+
    '<li><bold>Date : '+str(temp_date.iloc[i]['Date'])+
    '<li><bold>Provinces : '+str(temp_date.iloc[i]['Province/State'])+  
    '<li><bold>Confirmed : '+str(temp_date.iloc[i]['Confirmed'])+
    '<li><bold>Fatalities : '+str(temp_date.iloc[i]['Deaths'])+
    '<li><bold>Mortality Rate : '+str(temp_date.iloc[i]['Mortality Rate(Per 100)'])+
    '<li><bold>Active : '+str(temp_date.iloc[i]['Active'])+
    '<li><bold>Recovered : '+str(temp_date.iloc[i]['Recovered']),
    radius = int(temp_date.iloc[i]['Confirmed'])**1.1).add_to(m)
m




df_grouped_china.head(5)
df_grouped_china.loc[df_grouped_china['Date'] == '2020-04-08']




fig = go.Figure(data=[go.Bar(name='Confirmed Cases', 
                             x=df_grouped_china['Date'], y=df_grouped_china['Confirmed'])])

fig.update_layout(barmode='overlay',
                  legend=dict(
                    x=0,
                    y=1,
                    bgcolor="rgba(255,0,255,0.4)",
                    bordercolor="Black",
                    borderwidth=1),
                  font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
        ),
                  showlegend = True,
                  legend_title = '<b> China <b>',
                  title = 'Daily Confirmed Cases in China',
                 annotations = [dict(x = '2020-01-23', y = 643, 
                                    xref = 'x', yref = 'y', 
                                    text = 'Lockdown imposed in Wuhan(23th January)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 14,
                                     color = '#000000'),
                                    ax = -100, ay=-100),
                                dict(x = '2020-02-20', y = 75077, 
                                    xref = 'x', yref = 'y', 
                                    text = 'Lockdown extension decision(20th February)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 14,
                                     color = '#000000'),
                                    ax = -80, ay=-100),
                                 dict(x = '2020-01-11', y = 1, 
                                    xref = 'x', yref = 'y', 
                                    text = 'First official fatality(11th January)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 14,
                                     color = '#922b21'),
                                    ax = -180, ay=-180),
                                dict(x = '2020-04-08', y = 82809, 
                                    xref = 'x', yref = 'y', 
                                    text = 'Lockdown Ends in Wuhan(8th April)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 14,
                                     color = '#922b21'),
                                    ax = -50, ay=-80)
                               ])
fig.show()




df_italy_spain_us = df_full_table[df_full_table['Country/Region'].isin(['Italy', 'Spain', 'US'])]
# df_italy_spain_us




px.line(df_italy_spain_us, x='Date',
        y='Confirmed',
        color='Country/Region', 
       title = 'Confirmed cases of Italy, Spain & USA',
       height = 700, width = 900)









country_wise1 = df_full_group[df_full_group['Date'] == (df_full_group['Date'])].reset_index(drop=True)




df_china_spain_us = country_wise1[country_wise1['Country/Region'].isin(['China', 'Spain', 'US'])]
# df_china_spain_us




px.line(df_china_spain_us, x='Date',
        y='Confirmed',
        color='Country/Region', 
       title = 'Confirmed cases of China, Spain & USA',
       height = 700, width = 900)




# country_wise1




df_spain_uk_usa = country_wise1[country_wise1['Country/Region'].isin(['Spain', 'United Kingdom', 'US'])]
# df_spain_uk_usa




px.line(df_spain_uk_usa, x = 'Date',
       y = 'Confirmed', color = 'Country/Region',
       title = 'Confirmed cases of Spain, UK & USA', 
       height = 700, width = 900)




# df_full_table




#USA Data
df_us = df_full_table[df_full_table['Country/Region'] == 'US']
df_us

df_grouped_us = df_us.groupby('Date')[ 'Date', 'Confirmed', 'Deaths'].sum().reset_index()
# df_grouped_us




fig = px.line(df_grouped_us, x = 'Date', y = 'Confirmed',
             title = 'Confirmed cases of USA',
             color_discrete_sequence = ['#C70039'],
             height=700, width = 900)
fig.show()




fig = px.line(df_grouped_us, x = 'Date', y = 'Deaths',
             title = 'Death cases of USA',
             color_discrete_sequence = ['#581845'],
             height=700, width = 900)
fig.show()




df_us = country_wise1[country_wise1['Country/Region'] == 'US']
# df_us




px.line(df_us, x = 'Date',
       y = 'New cases',
       title = 'New cases cases of USA', 
       height = 700, width = 900)




df_usa_table.head(1)




df_usa_table.loc[:,["Confirmed","Deaths","Province_State"]].groupby(["Province_State"]).sum().sort_values("Confirmed",ascending=False).style.background_gradient(cmap='Blues',subset=["Confirmed"]).background_gradient(cmap='Reds',subset=["Deaths"])




df_usa_table.head(5)




#India Data
df_india = df_full_table[df_full_table['Country/Region'] == 'India']
df_india

df_grouped_india = df_india.groupby('Date')['Country/Region','Date', 'Confirmed', 'Deaths', 'Active'].sum().reset_index()
df_grouped_india




fig = px.line(df_grouped_india, x = 'Date', y = 'Confirmed',
             title = 'Confirmed cases of India',
             color_discrete_sequence = ['#27AE60'],
             height=700, width = 900)

              
fig.show()




# india_data = requests.get('https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise').json()
# df_india = pd.io.json.json_normalize(india_data['data']['statewise'])
# df_india = df_india.set_index('state')

# df_india["Mortality Rate (per 100)"] = np.round(100*df_india["deaths"]/df_india["confirmed"], 2)
# df_india




# #Calculate mortality rate and total
# total_india = df_india.sum()
# total_india.name = "Total numbers"
# df_total_india = pd.DataFrame(total_india, dtype=float).transpose()
# df_total_india["Mortality Rate (per 100)"] = np.round(100*df_total_india["deaths"]/df_total_india["confirmed"], 2)
# df_total_india.style.background_gradient(cmap='Reds', axis=1)




df_india.sort_values('confirmed', ascending = False).style.background_gradient(cmap='Blues', subset=['confirmed']).background_gradient(cmap='Reds', subset=['deaths']).background_gradient(cmap='Greens', subset=['recovered']).background_gradient(cmap='Oranges', subset=['active']).background_gradient(cmap='pink', subset=['Mortality Rate (per 100)'])




fig = go.Figure(data=[go.Scatter(name='Confirmed Cases', 
                             x=df_grouped_india['Date'], y=df_grouped_india['Confirmed'])])

fig.update_layout(barmode='overlay',
                  legend=dict(
                    x=0,
                    y=1,
                    bgcolor="rgba(255,0,255,0.4)",
                    bordercolor="Black",
                    borderwidth=1),
                  font=dict(
                    family="sans-serif",
                    size=12,
                    color="black"
        ),
                  showlegend = True,
                  legend_title = '<b> India <b>',
                  title = 'Daily Confirmed Cases in India',
                 annotations = [dict(x = '2020-03-25', y = 618, 
                                    xref = 'x', yref = 'y', 
                                    text = 'Phase 1 Lockdown imposed(25th March)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 14,
                                     color = '#000000'),
                                    ax = -80, ay=-150),
                               dict(x = '2020-04-15', y = 12403, 
                                    xref = 'x', yref = 'y', 
                                    text = 'Phase 2 Lockdown imposed(15th April)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 14,
                                     color = '#000000'),
                                    ax = -80, ay=-170),
                               dict(x = '2020-05-04', y = 47388, 
                                    xref = 'x', yref = 'y', 
                                    text = 'Phase 3 Lockdown imposed(4th May)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 14,
                                     color = '#000000'),
                                    ax = -80, ay=-150),
                               dict(x = '2020-05-18', y = 100172, 
                                    xref = 'x', yref = 'y', 
                                    text = 'Phase 4 Lockdown imposed(18th May)',
                                    showarrow=True, arrowhead=5,
                                     arrowwidth=2,
                                     font=dict(
                                     size = 14,
                                     color = '#000000'),
                                    ax = -80, ay=-150)])
fig.show()




df_grouped_india.head()




fig = make_subplots(rows = 1, cols = 2)

fig.add_scatter(x=df_grouped_india['Date'], y=df_grouped_india['Active'], row=1, col=1, name = 'Active Cases')
fig.add_bar(x=df_grouped_india['Date'], y=df_grouped_india['Deaths'], row=1, col=2, name = 'Fatalities')

fig.update_layout(height = 600, width = 800, title = 'Active & Fatality cases in India')

fig.show()




# temp_date = df_india[df_india['Date'] == max(df_india['Date'])]
# temp_date
# m = folium.Map(location=[19, 72], tiles='cartodbpositron',
#               min_zoom=3, max_zoom=5, zoom_start=3)
# for i in range(0, len(temp_date)):
#     folium.Circle(
#     location=[temp_date.iloc[i]['Lat'], temp_date.iloc[i]['Long']],
#     color='#FF5733 ',fill='crimson',
#     tooltip = '<li><bold>Country : '+str(temp_date.iloc[i]['Country/Region'])+
#     '<li><bold>Date : '+str(temp_date.iloc[i]['Date'])+
#     '<li><bold>Provinces : '+str(temp_date.iloc[i]['state'])+  
#     '<li><bold>Confirmed : '+str(temp_date.iloc[i]['confirmed'])+
#     '<li><bold>Fatalities : '+str(temp_date.iloc[i]['deaths'])+
#     '<li><bold>Mortality Rate : '+str(temp_date.iloc[i]['Mortality Rate(Per 100)'])+
#     '<li><bold>Active : '+str(temp_date.iloc[i]['active'])+
#     '<li><bold>Recovered : '+str(temp_date.iloc[i]['recovered']),
#     radius = int(temp_date.iloc[i]['Confirmed'])**1.1).add_to(m)
# m




#South Korea Data
df_south_korea = df_full_table[df_full_table['Country/Region'] == 'South Korea']
df_south_korea

df_grouped_south_korea = df_south_korea.groupby('Date')['Country/Region','Date', 'Confirmed', 'Deaths'].sum().reset_index()
df_grouped_south_korea




fig = px.line(df_grouped_south_korea, x = 'Date', y = 'Confirmed',
             title = 'Confirmed cases of India',
             color_discrete_sequence = ['#27BC20'],
             height=700, width = 900)

              
fig.show()

