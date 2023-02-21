#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

traffic_data = pd.read_csv('../input/accident.csv',
                           usecols=[0, 1, 11, 12, 13, 25, 26, 50, 51])
traffic_data = traffic_data.rename(
    columns={'ST_CASE':'case_id', 'STATE':'state', 'LATITUDE':'latitude',
             'LONGITUD':'longitude', 'DAY':'day', 'MONTH':'month', 'YEAR':'year',
             'DRUNK_DR':'drunk_drivers', 'FATALS':'fatalities'})
traffic_data['date'] = pd.to_datetime(traffic_data[['day', 'month', 'year']])
month_names = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July',
               8:'August', 9:'September', 10: 'October', 11:'October', 12:'December'}
traffic_data['month'] = traffic_data['month'].replace(month_names, regex=True)
traffic_data = traffic_data[['case_id', 'state', 'latitude', 'longitude', 'date', 'day',
                             'month', 'year', 'drunk_drivers', 'fatalities']]

us_states = np.asarray(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',                     'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',                     'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',                     'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA',                     'WA', 'WV', 'WI', 'WY'])




traffic_data['text'] = traffic_data['month'] + ' ' + traffic_data['day'].astype(str) +                       ', ' + traffic_data['fatalities'].astype(str) + ' Dead'

traffic_scale = [[0, 'rgb(181, 18, 18)'],[1, 'rgb(202,20,21)']]

data = [dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = traffic_data['longitude'],
        lat = traffic_data['latitude'],
        text = traffic_data['text'],
        mode = 'markers',
        marker = dict( 
            size = traffic_data['fatalities'] ** 0.5 * 5,
            opacity = 0.75,
            autocolorscale = False,
            colorscale = traffic_scale,
            showscale = False,
            cmin = 1,
            color = traffic_data['fatalities'],
            cmax = 10)
        )]

layout = dict(
        title = 'Traffic Fatalities by Latitude/Longitude in United States (2015)',
        geo = dict(
            scope = 'usa',
            projection = dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(250, 250, 250)',
            subunitwidth = 1,
            subunitcolor = 'rgb(217, 217, 217)',
            countrywidth = 1,
            countrycolor = 'rgb(217, 217, 217)',
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'
        ) )

figure = dict(data=data, layout=layout)
iplot(figure)




# traffic fatalities per state
traffic_perstate = np.asarray(traffic_data.groupby('state')['fatalities'].sum())

# state population estimates for July 2015 from US Census Bureau
# www.census.gov/popest/data/state/totals/2015/tables/NST-EST2015-01.csv
state_population = np.asarray([4858979, 738432, 6828065, 2978204, 39144818, 5456574,                               3590886, 945934, 646449, 20271272, 10214860, 1431603,                               1654930, 12859995, 6619680, 3123899, 2911641, 4425092,                               4670724, 1329328, 6006401, 6794422, 9922576, 5489594,                               2992333, 6083672, 1032949, 1896190,2890845, 1330608,                               8958013, 2085109, 19795791, 10042802, 756927, 11613423,                               3911338, 4028977, 12802503, 1056298, 4896146, 858469,                               6600299, 27469114, 2995919, 626042, 8382993, 7170351,                               1844128, 5771337, 586107])

# traffic fatalities per 100,000 people in state
traffic_percapita = traffic_perstate / state_population * 100000

traffic_scale = [[0, 'rgb(229, 243, 248)'],[1, 'rgb(0, 142, 194)']]

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = traffic_scale,
        showscale = False,
        locations = us_states,
        z = traffic_percapita,
        locationmode = 'USA-states',
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            )
        )]

layout = dict(
        title = 'Traffic Fatalities per 100,000 People in United States (2015)',
        geo = dict(
            scope = 'usa',
            projection = dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
            countrycolor = 'rgb(255, 255, 255)')
        )

figure = dict(data=data, layout=layout)
iplot(figure)




# traffic fatalities from drunk driving per state
traffic_datadrunk = traffic_data[traffic_data.drunk_drivers > 0]
drunk_perstate = np.asarray(traffic_datadrunk.groupby('state')['fatalities'].sum())

# traffic fatalities from drunk driving per 100,000 people in state
drunk_percapita = drunk_perstate / state_population * 100000

drunk_scale = [[0, 'rgb(250, 233, 233)'],[1, 'rgb(210, 42, 42)']]

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = drunk_scale,
        showscale = False,
        locations = us_states,
        z = drunk_percapita,
        locationmode = 'USA-states',
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            )
        )]

layout = dict(
        title = 'Traffic Fatalities from Drunk Driving per 100,000 People in United States (2015)',
        geo = dict(
            scope = 'usa',
            projection = dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
            countrycolor = 'rgb(255, 255, 255)')
        )

figure = dict(data=data, layout=layout)
iplot(figure)




x = 99
print(x)




traffic_data




traffic_data['date']




traffic_data = pd.read_csv('../input/accident.csv',
                           usecols=[0, 1, 11, 12, 13, 25, 26, 50, 51])




traffic_data




traffic_data = traffic_data.rename(
    columns={'ST_CASE':'case_id', 'STATE':'state', 'LATITUDE':'latitude',
             'LONGITUD':'longitude', 'DAY':'day', 'MONTH':'month', 'YEAR':'year',
             'DRUNK_DR':'drunk_drivers', 'FATALS':'fatalities'})




traffic_data




traffic_data[['day', 'month', 'year']]




0.45978 * 7




11.52 * 7 /40.6




us_states = np.asarray(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',                     'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',                     'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',                     'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA',                     'WA', 'WV', 'WI', 'WY'])




import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

traffic_data = pd.read_csv('../input/accident.csv',
                           usecols=[0, 1, 11, 12, 13, 25, 26, 50, 51])




traffic_data




us_states = np.asarray(['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA',                     'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',                     'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND',                     'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA',                     'WA', 'WV', 'WI', 'WY'])




us_states

