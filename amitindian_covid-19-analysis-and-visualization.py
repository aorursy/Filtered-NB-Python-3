#!/usr/bin/env python
# coding: utf-8



get_ipython().system(' pip install calmap')




import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import bokeh
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
output_notebook()
from bokeh.models import HoverTool

import warnings 
warnings.filterwarnings('ignore')


covid = pd.read_csv('../input/covid-19-data/covid_clean_complete_data.csv',parse_dates=['Date'])




# color pallette
c = '#393e46' # confirmed - grey
d = '#ff2e63' # death - red
r = '#30e3ca' # recovered - cyan
s = '#f8b400' # still infected - yellow




covid.info()          # dataframe info




covid.isnull().sum()             # checking for missing value




covid.describe()




# still infected = confirmed - (deaths - recovered)
covid['Actual Confirmed'] =covid['Confirmed']-covid['Deaths']-covid['Recovered'] 

# replacing Mainland china with China
covid['Country/Region']=covid['Country/Region'].replace('Mainland China','China') 

# filling missing values 
num_data =['Confirmed','Deaths','Recovered','Actual Confirmed']
covid[['Province/State']] = covid[['Province/State']].fillna('')
covid[num_data]=covid[num_data].fillna(0)




## create a seprate table for ship
ship = covid[covid['Province/State'].str.lower().str.contains('ship')]

## create a seprate table for china

china= covid[covid['Country/Region']=='China']

## create a seprate table for rest of the world

row = covid[covid['Country/Region']!='China']




#Add month and week
covid['Month']= covid['Date'].dt.month
covid['Week']= covid['Date'].dt.week
covid =covid[['Province/State','Country/Region','Lat','Long','Date','Confirmed','Deaths','Recovered','Actual Confirmed','Month','Week']]
# latest
covid_latest=covid[covid['Date']==max(covid['Date'])].reset_index()
china_latest= covid_latest[covid_latest['Country/Region']=='China']
row_latest= covid_latest[covid_latest['Country/Region']!='China']

#Groupby tabel 
covid_latest_group=covid_latest.groupby('Country/Region')['Confirmed','Deaths','Recovered','Actual Confirmed'].sum().reset_index()
china_latest_group=china_latest.groupby('Province/State')['Confirmed','Deaths','Recovered','Actual Confirmed'].sum().reset_index()
row_latest_group=row_latest.groupby('Country/Region')['Confirmed','Deaths','Recovered','Actual Confirmed'].sum().reset_index()




covid.head()




covid_group_latest= covid.groupby(['Country/Region','Province/State'])['Confirmed','Deaths','Recovered','Actual Confirmed'].max()
covid_group_latest.style.background_gradient(cmap='Reds')




covid_group_latest.head()




Current_Summary=covid_latest.groupby('Date')['Confirmed','Deaths','Recovered','Actual Confirmed'].sum().reset_index()
Current_Summary=Current_Summary[Current_Summary['Date']==max(Current_Summary['Date'])].reset_index(drop = True)
Current_Summary.style.background_gradient(cmap='Pastel1')




covid_latest.tail()




country_sum= covid_latest_group.sort_values(by='Confirmed', ascending=False)
country_sum = country_sum.reset_index(drop=True)
country_sum.style.background_gradient(cmap='Reds')




country_death=country_sum[country_sum['Deaths']>0][['Country/Region','Deaths']]
country_death.sort_values('Deaths',ascending = False).reset_index(drop=True).style.background_gradient(cmap='Reds')




country_no_recovered= country_sum[country_sum['Recovered']==0][['Country/Region','Confirmed','Deaths','Recovered']]
country_no_recovered.reset_index(drop = True).style.background_gradient(cmap = 'Reds')




Country_allcases_died= row_latest_group[row_latest_group['Confirmed']==
                                        row_latest_group['Deaths']] 
Country_allcases_died= Country_allcases_died[['Country/Region','Confirmed','Deaths']].sort_values('Confirmed',                 
                                    ascending =False)
Country_allcases_died.reset_index(drop = True)
Country_allcases_died.style.background_gradient(cmap ='Red')




country_recovered = row_latest_group[row_latest_group['Confirmed']==
                                     row_latest_group['Recovered']]
country_recovered = country_recovered[['Country/Region','Confirmed','Recovered']]
country_recovered = country_recovered.sort_values('Confirmed', ascending = False)
country_recovered = country_recovered.reset_index(drop = True)
country_recovered.style.background_gradient(cmap='Reds')




country_no_effected = row_latest_group[row_latest_group['Confirmed']==
                                row_latest_group['Deaths']+
                                row_latest_group['Recovered']]
country_no_effected = country_no_effected[['Country/Region', 'Confirmed', 'Deaths', 'Recovered']]
country_no_effected = country_no_effected.sort_values('Confirmed', ascending = False)
country_no_effected.reset_index(drop = True)
country_no_effected.style.background_gradient(cmap ='Reds')




china_data = china_latest_group[['Province/State', 'Confirmed', 'Deaths', 'Recovered','Actual Confirmed']]
china_data = china_data.sort_values(by='Confirmed' , 
                                    ascending = False)
china_data = china_data.reset_index(drop = True)  
china_data.style.background_gradient(cmap = 'Pastel1_r')




china_nocase_recovered = china_latest_group[china_latest_group['Recovered']==0]
china_nocase_recovered = china_nocase_recovered[['Province/State', 'Confirmed', 'Deaths', 'Recovered','Actual Confirmed']]
china_nocase_recovered = china_nocase_recovered.sort_values('Confirmed',
                                                            ascending = False)
china_nocase_recovered = china_nocase_recovered.reset_index(drop = True)
china_nocase_recovered




china_allcases_died = china_latest_group[china_latest_group['Confirmed']==
                                         china_latest_group['Deaths']]
china_allcases_died = china_allcases_died[['Province/State', 'Confirmed', 'Deaths', 'Recovered']]
china_allcases_died = china_allcases_died.sort_values('Confirmed', 
                                                      ascending = False)
china_allcases_died = china_allcases_died.reset_index(drop = True)
china_allcases_died




china_allcases_recovered = china_latest_group[china_latest_group['Confirmed']==
                                              china_latest_group['Recovered']]
china_allcases_recovered = china_allcases_recovered[['Province/State', 'Confirmed', 'Recovered']].set_index('Province/State')
china_allcases_recovered = china_allcases_recovered.sort_values('Confirmed', 
                                                                ascending = False).style.background_gradient(cmap = "Reds")
china_allcases_recovered




china_no_effected = china_latest_group[china_latest_group['Confirmed']==china_latest_group['Deaths']+
                                       china_latest_group['Recovered']]
china_no_effected = china_no_effected[['Province/State', 'Confirmed','Deaths', 'Recovered']].set_index('Province/State')
china_no_effected = china_no_effected.sort_values('Confirmed', ascending = False)
china_no_effected.style.background_gradient(cmap ='Reds')




covid.head()




temp = covid_latest.groupby('Date')['Recovered', 'Deaths', 'Actual Confirmed'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Actual Confirmed'],
                 var_name='Case', value_name='Count')
temp.head()




covid['Month'] = covid['Month'].replace(1,'Jan 20')
covid['Month'] = covid['Month'].replace(2,'Feb 20')
covid['Month'] = covid['Month'].replace(3,'Mar 20')
covid['Month'] = covid['Month'].replace(4,'Apr 20')

covid_Jan = covid[covid['Month']=='Jan 20']
covid_Feb = covid[covid['Month']=='Feb 20']
covid_Mar = covid[covid['Month']=='Mar 20']
covid_Apr = covid[covid['Month']=='Apr 20']




Date_wise=covid.groupby('Date')['Confirmed','Deaths','Recovered','Actual Confirmed'].sum().reset_index()
Date_wise.head()




Week_wise=covid.groupby('Week')['Confirmed','Deaths','Recovered','Actual Confirmed'].sum().reset_index()
Week_wise




Month_wise=covid.groupby('Month')['Confirmed','Deaths','Recovered','Actual Confirmed'].sum().reset_index()
Month_wise = Month_wise.sort_values('Month', ascending=False)
Month_wise




from bokeh.models.annotations import Title




p = figure(height = 300, width = 600)
p.triangle(x = covid_Jan['Actual Confirmed'], y = covid_Jan['Deaths'],color ='red',line_width=2)     
p.xaxis.axis_label = "Number of Cases Confirmed"
p.yaxis.axis_label = "Number of Cases Deaths"
p.title.text='Month Of Jan 20'
p.add_tools(HoverTool())
show(p)




p = figure(height = 300, width = 600)
p.triangle(x = covid_Feb['Actual Confirmed'], y = covid_Feb['Deaths'],color ='red',line_width=2)     
p.xaxis.axis_label = "Number of Cases Confirmed"
p.yaxis.axis_label = "Number of Cases Deaths"
p.title.text='Month Of Feb 20'
p.add_tools(HoverTool())
show(p)




p = figure(height = 300, width = 600)
p.triangle(x = covid_Mar['Actual Confirmed'], y = covid_Mar['Deaths'],color ='red',line_width=3)     
p.xaxis.axis_label = "Number of Cases Confirmed"
p.yaxis.axis_label = "Number of Cases Deaths"
p.title.text='Month Of Mar 20'
p.add_tools(HoverTool())
show(p)




p = figure(height = 300, width = 600)
p.triangle(x = covid_Apr['Actual Confirmed'], y = covid_Apr['Deaths'],color ='red',line_width=2)     
p.xaxis.axis_label = "Number of Cases Confirmed"
p.yaxis.axis_label = "Number of Cases Deaths"
p.title.text='Month Of Apr 20'
p.add_tools(HoverTool())
show(p)




plt.figure(figsize=(8,6))
corre_data = covid_latest.iloc[:,5:10]
corr= corre_data.corr()
sns.heatmap(corr,annot =True,cmap="YlGnBu",linewidths =0.2)




plt.figure(figsize=(10,4))
sns.barplot(x = 'Month', y = 'Actual Confirmed', data = Month_wise,
            palette = 'hls',
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'   
            )
plt.show()




plt.figure(figsize=(16,4))
sns.barplot(x = 'Week', y = 'Actual Confirmed', data = Week_wise,
            palette = 'hls',
            capsize = 0.05,             
            saturation = 8,             
            errcolor = 'gray', errwidth = 2,  
            ci = 'sd'   
            )
plt.show()

