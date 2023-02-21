#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)




dn = pd.read_csv('../input/NationalNames.csv',index_col='Id')
ds = pd.read_csv('../input/StateNames.csv',index_col='Id')




dn = dn.set_index(['Name','Year','Gender']).unstack().fillna(0).astype(int)
dn.columns = ['CountF','CountM']
dn['CountTotal'] = dn.CountF + dn.CountM
dn['CountYear'] = dn.groupby(level=['Year'])['CountTotal'].transform('sum')
dn['Popularity'] = 1000*dn.CountTotal.values / dn.CountYear.values #babies per thousand
dn['Ambiguity'] = dn[['CountF','CountM']].min(axis=1).values/dn.CountTotal.values
dn['AmbiguityWeighted'] = dn.Ambiguity * dn.Popularity
dn.head()




amb = dn.groupby(level='Year')['AmbiguityWeighted'].sum()/1000
amb.plot()
plt.title('Gender ambiguity of baby names');




ds = ds.set_index(['Name','Year','State','Gender']).unstack().fillna(0).astype(int)
ds.columns = ['CountF','CountM']
ds['CountTotal'] = ds.CountF + ds.CountM
ds['CountYearState'] = ds.groupby(level=[1,2])['CountTotal'].transform('sum')
ds['Popularity'] = 1000*ds.CountTotal.values / ds.CountYearState.values
ds['Ambiguity'] = ds[['CountF','CountM']].min(axis=1).values/ds.CountTotal.values
ds['AmbiguityWeighted'] = ds.Ambiguity * ds.Popularity
ds.head()




ambs = ds.groupby(level=['Year','State'])         ['AmbiguityWeighted'].sum()         .unstack().fillna(0)/1000




ambs.head(2)




ambs.loc[2004,'KY'] = np.NaN
ambs.loc[[1989,1990],'DC'] = np.NaN
ambs = ambs.interpolate()




fig, ax = plt.subplots(figsize=(12,6))
amb.plot(ax=ax,linewidth=5,zorder=100)
ambs.plot(color='#d7472f',alpha=0.6, ax=ax)
ax.legend(['Overall','States'])
ax.set_xlim(1920,2015)
ax.set_xticks(np.arange(1920,2020,10))
ax.set_title('Gender ambiguity over time');




def ambiguity_map(year):
    minyear = max(year-5, ambs.index.min())
    maxyear = min(year+5, ambs.index.max())
    df = pd.DataFrame(ambs.loc[minyear:maxyear,:].mean(),columns=['Ambiguity']).reset_index()
    data = [ dict(
            type='choropleth',
            autocolorscale = True,
            locations = df['State'],
            z = df['Ambiguity'].astype(float),
            zmax = ambs.max().max(),
            zmin = ambs.min().min(),
            locationmode = 'USA-states',
            text = df['State'],
            marker = dict(
                line = dict (
                    color = 'rgb(255,255,255)',
                    width = 1
                ) ),
            colorbar = dict(
                title = "Gender Ambiguity")
            ) ]

    layout = dict(
            title = 'Gender Ambiguity by state {}-{}'.format(minyear, maxyear),
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showlakes = True,
                lakecolor = 'rgb(255, 255, 255)'),
                 )

    fig = dict( data=data, layout=layout )
    iplot( fig, filename='d3-cloropleth-map' )




ambiguity_map(1925)




ambiguity_map(2009)




ds.loc[pd.IndexSlice[:,list(range(1920,1930)),
       ['MS','TX','AL','GA','AR','TN']],'AmbiguityWeighted']\
  .groupby(level='Name').mean().sort_values(ascending=False).head(20)




names = [n for n in _.index if n.endswith('ie')]
names




df = ds.loc[pd.IndexSlice[names,tuple(range(1920,1930)),:],'Popularity'].groupby(level='State').sum().reset_index()
df.Popularity/=10 # mean over 10 years
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = df['State'],
        z = df['Popularity'].astype(float),
        zmax = df.Popularity.max(),
        zmin = 0.,
        locationmode = 'USA-states',
        text = df['State'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 1
            ) ),
        colorbar = dict(
            title = "Popularity<br>babies per thousand")
        ) ]

layout = dict(
        title = 'Popularity of "-ie" names by state in 1920-1930<br>'+', '.join(names),
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )

fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )




fig, ax = plt.subplots(1,2,figsize=(12,5))
fig.suptitle(', '.join(names),fontsize='large')
dn.loc[names].groupby(level='Year')['CountM','CountF'].sum().plot(ax=ax[0]);
dn.loc[names].groupby(level='Year')['Popularity'].sum().plot(ax=ax[1]).legend();




n = dn.groupby(level='Name')['AmbiguityWeighted'].max().sort_values(ascending=False)
n.head(20)




df = dn.loc[list(n.head(12).index),'AmbiguityWeighted'].unstack().T.fillna(0)
data = []
for col in df.columns:
    data.append(
        go.Scatter(
        x=df.index,
        y=df[col].values,
        name=col,
        text=col
        ))

layout = go.Layout(
    title='Popular gender ambiguous names over time<br>(hover over line to see the name)',
    hovermode= 'closest',
    xaxis=dict(title='Year'),
    yaxis=dict(title='Ambiguity * Popularity')
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='gender-ambiguous-names')




df = dn.loc[pd.IndexSlice[:,2014],:]        .sort_values(by='AmbiguityWeighted',ascending=False).head(100).reset_index()
df1 = df.head(30)
df2 = df.tail(70)
trace1 = go.Scatter(
    x=df1.CountF,
    y=df1.CountM,
    marker = dict(color=df1.CountF.values/(df1.CountTotal).values,
                  colorscale = 'Viridis'),
    text = df1.Name,
    mode='markers+text',
    textposition = 'top center'
)
trace2 = go.Scatter(
    x=df2.CountF,
    y=df2.CountM,
    marker = dict(color=df2.CountF.values/(df2.CountTotal).values,
                  colorscale = 'Viridis'),
    text = df2.Name,
    mode='markers',
)
line = [df.CountTotal.min()/2, df.CountTotal.max()/2]
trace3 = go.Scatter(
    x = line, y=line, 
    mode='lines',
    line=dict(color='rgba(0,0,0,0.1)',width=1))
data = [trace1, trace2,trace3]
layout = go.Layout(
    title='Gender neutral names of 2014',
    autosize=False,
    width=800,
    height=800,
    showlegend=False,
    hovermode='closest',
    xaxis=dict(type='log',
               title='Count of girls'),
    yaxis=dict(type='log',
               title='Count of boys'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='style-annotation')

