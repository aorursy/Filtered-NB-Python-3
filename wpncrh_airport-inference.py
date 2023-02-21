#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.graph_objs import *

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier




names = ['id',
'Name' ,
'City' ,
'Country',
'Airport ID',
'IATA' ,
'lat', 
'lon' ,
'Altitude',
'Timezone' ,
'DST' ,
'Tz' ,
'Type',
'Source']




stations = pd.read_csv('../input/airports-extended.csv', names = names)




stations.head(2)




airports = stations[stations['Type'] =='airport']
train_stations = stations[stations['Type'] =='station']
ferries = stations[stations['Type'] =='port']
other = stations[stations['Type'] =='unknown']




airports.shape




train_stations.shape




airports.Country.value_counts().head()




# Find counts of airports by country.
airports_country_counts = airports.Country.value_counts()
airports_country_counts_df = pd.DataFrame(airports_country_counts).reset_index()
airports_country_counts_df.columns = ['Country', 'Count']
airports_country_counts_df.tail()




# Do the same for train stations:
train_country_counts = train_stations.Country.value_counts()
train_country_counts_df = pd.DataFrame(train_country_counts).reset_index()
train_country_counts_df.columns = ['Country', 'Count']
train_country_counts_df.head()




number_countries_to_show = 10

trace1 = go.Bar(
    x=airports_country_counts_df[0:number_countries_to_show].Country,
    y=airports_country_counts_df[0:number_countries_to_show].Count,
    name='Airports'
)
trace2 = go.Bar(
    x=airports_country_counts_df[0:number_countries_to_show].Country,
    y=train_country_counts_df[0:number_countries_to_show].Count,
    name='Train stations'
)

data = [trace1, trace2]
layout = go.Layout(
    title = 'Airports and Train Station Counts <br> (by top 10 airport nations)',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)





counts_df = pd.merge(train_country_counts_df, airports_country_counts_df, 
             left_on='Country', 
             right_on = 'Country',
             how='inner')
counts_df.columns = ['Country','Trains','Airports']
counts_df['train/plane_ratio'] = counts_df['Trains']/counts_df['Airports']
counts_df.sort_values(by='train/plane_ratio')[0:10]                                                            




trace1 = go.Bar(
    x=counts_df.Country,
    y=counts_df['train/plane_ratio'],
    name='Ratio'
)


data = [trace1]
layout = go.Layout(
    title = 'Airports and Train Station Counts <br> (by top 10 airport nations)',
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)




conditions = [
    (stations['Type'] == 'airport'),
    (stations['Type'] == 'station'),
    (stations['Type'] == 'port'),
    (stations['Type'] == 'unknown')]

choices = ['blue', 'red', 'green','black']
stations['color'] = np.select(conditions, choices, default='black')




data = [ dict(
        type = 'scattergeo',
#         locations = country_counts_df['Country'],
        lat = stations.lat,
        lon = stations.lon,
#         z = airports['lon'],
        text = stations['Country']+stations['Name'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = True,
        reversescale = True,
        marker = dict(
            color = stations.color
        )
         )]

layout = dict(
    title= 'Locations of stations, colored by type <br> (zoom and scroll to adjust view)',
    geo = dict(
        scope='usa',
        showframe = True,
        showcoastlines = True,
        showcountries=True,
        projection = dict(
            type = 'equirectangular'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )




# drop color variable from before:
stations = stations.drop('color', 1)
# split into known and unknown station type
known = stations[stations['Type'] !='unknown']
prediction = stations[stations['Type'] =='unknown']




known.head()




training_parent = known[['lat','lon','Altitude','Type']]
prediction_parent = prediction[['lat','lon','Altitude']]




x_train, x_test, y_train, y_test = train_test_split(training_parent[['lat','lon','Altitude']], 
                                                    training_parent[['Type']],
                                                    test_size=0.33, random_state=42)




print('Train shape is: ',x_train.shape,' and the shape of the test set is: ',x_test.shape) 




# convert to np array
np_x_train = x_train.values
np_y_train = y_train.values




# train
model = XGBClassifier()
model.fit(np_x_train, np_y_train.ravel())
print(model)




# predict
y_pred = model.predict(x_test.values)




y_pred_df = pd.Series(y_pred)




print('Predictions:',y_pred_df.value_counts())
print('Actual test set distribution:', y_test.Type.value_counts())




from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)




# predict on the unknowns
prediction_final = model.predict(prediction_parent.values)




fps = pd.Series(prediction_final)
end_result = prediction.reset_index()
end_result['prediction'] = fps
end_result.head(50)




end_result[end_result['prediction'] == 'station'].head()











