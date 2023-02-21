#!/usr/bin/env python
# coding: utf-8



# Packages
import pandas as pd
import numpy as np
import scipy as sp
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 




df = pd.read_csv("../input/vgsales.csv")
df[:10]




df.info()




df.shape




df.describe()




# Platform
df.Platform = df.Platform.astype('category')
df.Platform.describe()




platform_count = df.groupby('Platform', axis=0).count().reset_index()[['Platform','Name']].sort_values(by = "Name", ascending=True)




# Game counts by platform

import plotly.graph_objs as go

layout = go.Layout(
    title='Total Release by Platforms',
    yaxis=dict(
        title='Platform'
    ),
    xaxis=dict(
        title='Count'
    ),
    height=600, width=600
)

trace = go.Bar(
            x=platform_count.Name,
            y=platform_count.Platform,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
iplot(fig, show_link=False)




year_count = df.groupby('Year', axis=0).count().reset_index()[['Year','Name']]
year_count.Year = year_count.Year.astype('int')

# remove data after 2016
year_count = year_count[year_count.Year <= 2016]




trace = go.Scatter(
    x = year_count.Year,
    y = year_count.Name,
    mode = 'lines',
    name = 'lines'
    
)


layout = go.Layout(
    title='Release by Year',
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Year'
    ),
    height=600, width=600
)

fig = go.Figure(data=[trace], layout=layout)
iplot(fig)




genre_count = df.groupby('Genre', axis=0).count().reset_index()[['Genre','Name']].sort_values(by = "Name", ascending=True)
layout = go.Layout(
    title='Releases by Genre',
   
    xaxis=dict(
        title='Releases'
    ),
    height=400, width=600
)

trace = go.Bar(
            x=genre_count.Name,
            y=genre_count.Genre,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
iplot(fig)




publisher_count = df.groupby('Publisher', axis=0).count().reset_index()[['Publisher','Name']].sort_values(by = "Name", ascending=True)
publisher_count = publisher_count.tail(n=30)
layout = go.Layout(
    title='Release by Publisher (Top 30)',

    xaxis=dict(
        title='Releases'
    ),
    height=700, width=750,
    margin=go.Margin(
        l=300,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)

trace = go.Bar(
            x=publisher_count.Name,
            y=publisher_count.Publisher,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
iplot(fig)




publisher_sales = df.groupby('Publisher', axis=0).sum().reset_index()[['Publisher','Global_Sales']].sort_values(by = "Global_Sales", ascending=True)
publisher_sales = publisher_sales.tail(n=30)

layout = go.Layout(
    title='Sales by Publisher (Top 30)',

    xaxis=dict(
        title='Sales (in Millions)'
    ),
    height=700, width=700,
    margin=go.Margin(
        l=300,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)

trace = go.Bar(
            x=publisher_sales.Global_Sales,
            y=publisher_sales.Publisher,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
iplot(fig)




new_df = df
new_df['Game_Count'] = 1
new_df = new_df.groupby(['Publisher']).sum().reset_index()[['Publisher', 'Global_Sales','Game_Count']]
new_df['Revenue_per_game'] = new_df.Global_Sales/new_df.Game_Count

new_df = new_df.sort_values(by = "Revenue_per_game", ascending=True).                            tail(n=30)
layout = go.Layout(
    title='Revenue_per_game by Publisher (Top 30)',

    xaxis=dict(
        title='Revenue_per_game (in Millions)'
    ),
    height=700, width=700,
    margin=go.Margin(
        l=250,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)

trace = go.Bar(
            x=new_df.Revenue_per_game,
            y=new_df.Publisher,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
iplot(fig)




sales_by_genre = df.groupby(['Genre','Name'], axis = 0).sum().reset_index()[['Genre','Name','Global_Sales']]




import random
from numpy import * 
genres = sales_by_genre.Genre.unique()
traces = []
c = ['hsl('+str(h)+',50%'+',50%)' for h in linspace(0, 360, len(genres))]

for i in range(len(genres)):
    genre = genres[i]
    df_genre = sales_by_genre[sales_by_genre.Genre == genre]
    trace = go.Box(
        y=np.array(df_genre.Global_Sales),
        name=genre,
        boxmean=True,
        marker={'color': c[i]}
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Genre (A lot of outliers)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    ),
    height=700, width=700,
    margin=go.Margin(
        l=100,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)
    

fig = go.Figure(data=traces, layout=layout)
iplot(fig)




# The outliers are like:
df.groupby(['Genre','Name'], axis = 0).         sum()[['Global_Sales']].         sort_values(by="Global_Sales", ascending = False).         reset_index()[:10]




# After delete outlier

PERCENTAGE = 0.95
traces = []

for i in range(len(genres)):
    genre = genres[i]
    df_genre = sales_by_genre[sales_by_genre.Genre == genre]
    df_genre = df_genre[df_genre.Global_Sales < df_genre.Global_Sales.quantile(PERCENTAGE)]
    
    trace = go.Box(
        y=np.array(df_genre.Global_Sales),
        name=genre,
        boxmean=True,
        marker={'color': c[i]}
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Genre (TOP 1% games)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    ),
    height=700, width=700,
    margin=go.Margin(
        l=100,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)
    

fig = go.Figure(data=traces, layout=layout)
iplot(fig)




PERCENTAGE = 0.99
traces = []

for i in range(len(genres)):
    genre = genres[i]
    df_genre = sales_by_genre[sales_by_genre.Genre == genre]
    df_genre = df_genre[df_genre.Global_Sales > df_genre.Global_Sales.quantile(PERCENTAGE)]
    
    trace = go.Box(
        y=np.array(df_genre.Global_Sales),
        name=genre,
        boxmean=True,
        marker={'color': c[i]}
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Genre (TOP 1% games)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    ),
    height=700, width=700,
    margin=go.Margin(
        l=100,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)
    

fig = go.Figure(data=traces, layout=layout)
iplot(fig)




top10_publishers = np.array(df.groupby('Publisher', axis=0).sum().                           reset_index()[['Publisher','Global_Sales']].                           sort_values(by = "Global_Sales", ascending=True).                           tail(n=10)['Publisher'])

top10_df = df[[pub in top10_publishers for pub in df.Publisher]]
sales_by_publisher = top10_df.groupby(['Publisher','Name']).sum().reset_index()[['Publisher','Name','Global_Sales']]




PERCENTAGE = 0.9
traces = []

for i in range(len(top10_publishers)):
    publisher = top10_publishers[i]
    df_pub = sales_by_publisher[sales_by_publisher.Publisher == publisher]
    df_pub = df_pub[df_pub.Global_Sales < df_pub.Global_Sales.quantile(PERCENTAGE)]
    
    trace = go.Box(
        y=np.array(df_pub.Global_Sales),
        name=publisher,
        boxmean=True,
        marker={'color': c[i]}
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Publisher (Majority Games)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    ),
    height=700, width=700,
    margin=go.Margin(
        l=100,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)
    

fig = go.Figure(data=traces, layout=layout)
iplot(fig)




PERCENTAGE = 0.95
traces = []

for i in range(len(top10_publishers)):
    publisher = top10_publishers[i]
    df_pub = sales_by_publisher[sales_by_publisher.Publisher == publisher]
    df_pub = df_pub[df_pub.Global_Sales > df_pub.Global_Sales.quantile(PERCENTAGE)]
    
    trace = go.Box(
        y=np.array(df_pub.Global_Sales),
        name=publisher,
        boxmean=True,
        marker={'color': c[i]},
        boxpoints = 'all'
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Publisher (TOP 5% Games)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    ),
    height=700, width=700,
    margin=go.Margin(
        l=100,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)
    

fig = go.Figure(data=traces, layout=layout)
iplot(fig)




sales_by_year = df.groupby('Year', axis=0).sum().reset_index()[['Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']]
sales_by_year.Year = sales_by_year.Year.astype('int')




sales_by_year = sales_by_year[sales_by_year.Year <= 2016]




trace_Global = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.Global_Sales,
    mode = 'none',
    name = 'Global_Sales',
    fill='tonexty',
)

trace_NA = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.NA_Sales,
    mode = 'none',
    fill='tonexty',
    name = 'NA_Sales'
)

trace_EU = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.EU_Sales,
    mode = 'none',
    fill='tonexty',
    name = 'EU_Sales'
)

trace_JP = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.JP_Sales,
    mode = 'none',
    fill='tonexty',
    name = 'JP_Sales'
)

trace_Other = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.Other_Sales,
    mode = 'none',
    fill='tozeroy',
    name = 'Other_Sales'
)



layout = go.Layout(
    title='Sales by Region',

    xaxis=dict(
        title='Year'
    ),
    yaxis=dict(
        title='Sales (in Millions)'
    ),
    
    height=700, width=800,
    margin=go.Margin(
        l=100,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)


fig = go.Figure(data=[trace_Other, trace_JP, trace_EU, trace_NA, trace_Global], layout=layout)
iplot(fig)




# Get list of unique genres
genres = np.sort(df.Genre.unique())[::-1]

def get_traces(df, region):
    regional_df = df.groupby(['Genre','Year'], axis=0).sum().reset_index()[['Genre','Year', region]]
    years = list(range(1980,2018))
    
    temp_dict = {}
    for genre in genres:
        temp_dict[genre] = {}
        for year in years:
            try:
                temp_value = round(np.array(regional_df[(regional_df.Genre == genre) & 
                                   (regional_df.Year == year)][region])[0],2)
            except:
                temp_value = 0
            temp_dict[genre][year] = temp_value
    
    traces = []
    for genre in genres:
        trace = go.Bar(
            x = years,
            y = list(temp_dict[genre].values()),
            name=genre
        )
        traces.append(trace)
    
    return traces




data = get_traces(df, 'Global_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in Global',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        ),
    
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_traces(df, 'NA_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in North America',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        ),
    
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_traces(df, 'JP_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in Japan',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        ),
    
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_traces(df, 'EU_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in Europe',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        ),
    
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_traces(df, 'Other_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in Other (not JP, NA, EU)',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        ),
    
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




def get_percent_traces(df, region):
    temp_df = df.groupby(['Year','Genre'], axis=0).sum()[[region]]
    df_pcts = temp_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    df_pcts = df_pcts.reset_index()
    regional_df = df_pcts[df_pcts.Year < 2017] 
    
    years = list(range(1980,2018))
    
    temp_dict = {}
    for genre in genres:
        temp_dict[genre] = {}
        for year in years:
            try:
                temp_value = round(np.array(regional_df[(regional_df.Genre == genre) & 
                                   (regional_df.Year == year)][region])[0],2)
            except:
                temp_value = 0
            temp_dict[genre][year] = temp_value
    
    
    traces = []
    for genre in genres:
        trace = go.Bar(
            x = years,
            y = list(temp_dict[genre].values()),
            name=genre
        )
        traces.append(trace)
    
    return traces




data = get_percent_traces(df, 'Global_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in Global',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        ),
    
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_percent_traces(df, 'NA_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in North America',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_percent_traces(df, 'JP_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in Japan',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        ),
    
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_percent_traces(df, 'EU_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in Europe',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_percent_traces(df, 'Other_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in Other regions',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        ),
    
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)




len(df.Publisher.unique())




# Prefered genres of top-5-sale publishers
genres = genres[::-1]

def get_traces_genre_publisher(region):
    top5_publishers = np.array(df.groupby('Publisher', axis=0).sum().                               reset_index()[['Publisher', 'Global_Sales']].                               sort_values(by = 'Global_Sales', ascending=True).                               tail(n=5)['Publisher'])

    top5_df = df[[pub in top5_publishers for pub in df.Publisher]]
    top5_genre_df = top5_df.groupby(['Publisher','Genre']).sum().reset_index()[['Publisher','Genre',region]]

    traces = []
    for i in list(range(len(top5_publishers))):
        publisher = top5_publishers[i]
        temp_df = top5_genre_df[top5_genre_df.Publisher == publisher]
        
       

        trace = go.Bar(
            x = genres,
            y = np.array(temp_df[region]),
            name=publisher
        )
        traces.append(trace)

    return traces




data = get_traces_genre_publisher('Global_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'Global Sales by Genre and Publisher',
        height=700, width=800,
        margin=go.Margin(
            l=100,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)




# Top 5 games of Nintendo
df[df.Publisher == 'Nintendo'].sort_values(by = 'Global_Sales', ascending=False)[['Publisher','Name','Global_Sales']][:5]




# Top 5 games of EA
df[df.Publisher == 'Electronic Arts'].sort_values(by = 'Global_Sales', ascending=False)[['Publisher','Name','Global_Sales']][:5]




# Top 5 games of Activision
df[df.Publisher == 'Activision'].sort_values(by = 'Global_Sales', ascending=False)[['Publisher','Name','Global_Sales']][:5]




data = get_traces_genre_publisher('NA_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'North America - Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_traces_genre_publisher('JP_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'Japan - Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_traces_genre_publisher('EU_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'Europe - Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)




data = get_traces_genre_publisher('Other_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'Other Regions - Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig)




img = Image.open("sales-by-region.png")

draw = ImageDraw.Draw(img)
img.save('sales-by-region.png')

PImage("sales-by-region.png")

