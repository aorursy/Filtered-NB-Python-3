#!/usr/bin/env python
# coding: utf-8



# import additional packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




data_terrorism = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,2,3,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,30,32,34,36,38,39,40,42,44,46,47,48,50,52,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,80,81,83,85,87,89,91,93,95,97,98,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,116,117,118,119,120,122,124,125,126,127,128,130,131,132,133,134])
data_terrorism.head()




# Number of observations per column
data_terrorism.count()

# et cetera...




# Group data per year for the world
terror_peryear_world = np.asarray(data_terrorism.groupby('iyear').iyear.count())
terror_years = np.arange(1970, 2016)

# Explore in table iyear and years (is het mogelijk did in één tabel te krijgen??)
print(terror_peryear_world)
print(terror_years)




# Create graph for the wold

trace0 = [go.Scatter(
         x = terror_years,
         y = terror_peryear_world,
         mode = 'lines',
         line = dict(
             color = 'rgb(240, 140, 45)',
             width = 3),
        name = 'World'
         )]

layout = go.Layout(
         title = 'Terrorist Attacks by Year for the world (1970-2015)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
        yaxis = dict(
             range = [0.1, 17500],
             showline = True,
             showgrid = False)
         )

figure = dict(data = trace0, layout = layout)
iplot(figure)




# create distinct lines for the different regions
data_north_america = data_terrorism[(data_terrorism.region == 1) |  (data_terrorism.region == 2)]
data_asia = data_terrorism[(data_terrorism.region == 4) | (data_terrorism.region == 5) | (data_terrorism.region == 6) | (data_terrorism.region == 7)]
data_oceania = data_terrorism[(data_terrorism.region == 12)]
data_europe = data_terrorism[(data_terrorism.region == 8) | (data_terrorism.region == 9)]
data_south_america = data_terrorism[(data_terrorism.region == 3)]
data_middle_east_n_africa = data_terrorism[(data_terrorism.region == 10)]
data_sub_africa = data_terrorism[(data_terrorism.region == 11)]

peryear_north_america = np.asarray(data_north_america.groupby('iyear').iyear.count())
peryear_asia = np.asarray(data_asia.groupby('iyear').iyear.count())
peryear_oceania = np.asarray(data_oceania.groupby('iyear').iyear.count())
peryear_europe = np.asarray(data_europe.groupby('iyear').iyear.count())
peryear_south_america = np.asarray(data_south_america.groupby('iyear').iyear.count())
peryear_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('iyear').iyear.count())
peryear_sub_africa = np.asarray(data_sub_africa.groupby('iyear').iyear.count())




trace1 = go.Scatter(                             
         x = terror_years,
         y = peryear_north_america,
         mode = 'lines',
         line = dict(
             color = 'rgb(140, 140, 45)',
             width = 3),
        name = 'North- and Central America '
         )
trace2 = go.Scatter(                             
         x = terror_years,
         y = peryear_asia,
         mode = 'lines',
         line = dict(
             color = 'rgb(240, 40, 45)',
             width = 3),
        name = 'Asia'
         )
trace3 = go.Scatter(                             
         x = terror_years,
         y = peryear_oceania,
         mode = 'lines',
         line = dict(
             color = 'rgb(120, 120,120)',
             width = 3),
        name = 'Oceania'
         )
trace4 = go.Scatter(                             
         x = terror_years,
         y = peryear_europe,
         mode = 'lines',
         line = dict(
             color = 'rgb(0, 50, 72)',
             width = 3),
        name = 'Europe'
         )
trace5 = go.Scatter(                             
         x = terror_years,
         y = peryear_south_america,
         mode = 'lines',
         line = dict(
             color = 'rgb(27, 135 , 78)',
             width = 3),
        name = 'South America'
         )
trace6 = go.Scatter(                             
         x = terror_years,
         y = peryear_middle_east_n_africa,
         mode = 'lines',
         line = dict(
             color = 'rgb(230, 230, 230)',
             width = 3),
        name = 'Middle East and North Africa'
         )
trace7 = go.Scatter(                             
         x = terror_years,
         y = peryear_sub_africa,
         mode = 'lines',
         line = dict(
             color = 'rgb(238, 133, 26)',
             width = 3),
        name = 'Sub Saharan Africa'
         )

layout = go.Layout(
         title = 'Terrorist Attacks by Year per region (1970-2015)',
         xaxis = dict(
             rangeslider = dict(thickness = 0.05),
             showline = True,
             showgrid = False
         ),
         yaxis = dict(
             range = [0.1, 7500],
             showline = True,
             showgrid = False)
         )

data = [trace1, trace2, trace3, trace4, trace5, trace6, trace7]

figure = dict(data = data, layout = layout)
iplot(figure)




# create arrays with the values 0 and 1

crit1_north_america = np.asarray(data_north_america.groupby('crit1').crit1.count())
crit1_asia = np.asarray(data_asia.groupby('crit1').crit1.count())
crit1_oceania = np.asarray(data_oceania.groupby('crit1').crit1.count())
crit1_europe = np.asarray(data_europe.groupby('crit1').crit1.count())
crit1_south_america = np.asarray(data_south_america.groupby('crit1').crit1.count())
crit1_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('crit1').crit1.count())
crit1_sub_africa = np.asarray(data_sub_africa.groupby('crit1').crit1.count())

crit2_north_america = np.asarray(data_north_america.groupby('crit2').crit2.count())
crit2_asia = np.asarray(data_asia.groupby('crit2').crit2.count())
crit2_oceania = np.asarray(data_oceania.groupby('crit2').crit2.count())
crit2_europe = np.asarray(data_europe.groupby('crit2').crit2.count())
crit2_south_america = np.asarray(data_south_america.groupby('crit2').crit2.count())
crit2_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('crit2').crit2.count())
crit2_sub_africa = np.asarray(data_sub_africa.groupby('crit2').crit2.count())

crit3_north_america = np.asarray(data_north_america.groupby('crit3').crit3.count())
crit3_asia = np.asarray(data_asia.groupby('crit3').crit3.count())
crit3_oceania = np.asarray(data_oceania.groupby('crit3').crit3.count())
crit3_europe = np.asarray(data_europe.groupby('crit3').crit3.count())
crit3_south_america = np.asarray(data_south_america.groupby('crit3').crit3.count())
crit3_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('crit3').crit3.count())
crit3_sub_africa = np.asarray(data_sub_africa.groupby('crit3').crit3.count())




# Create the arrays for the graphs per region per criterium

regions = ['North- and Central-America', 'Asia', 'Oceania', 'Europe', 'South-America', 'Middle-East and North-Africa', 'Sub-saharan Africa']

# Create array for yes       --> crit2 oceania = 0
crit1_yes = np.append(crit1_north_america[1], crit1_asia[1])
crit1_yes = np.append(crit1_yes,crit1_oceania[1])
crit1_yes = np.append(crit1_yes,crit1_europe[1])
crit1_yes = np.append(crit1_yes,crit1_south_america[1])
crit1_yes = np.append(crit1_yes,crit1_middle_east_n_africa[1])
crit1_yes = np.append(crit1_yes,crit1_sub_africa[1])
crit2_yes = np.append(crit2_north_america[1], crit2_asia[1])
crit2_yes = np.append(crit2_yes, 0)
crit2_yes = np.append(crit2_yes,crit2_europe[1])
crit2_yes = np.append(crit2_yes,crit2_south_america[1])
crit2_yes = np.append(crit2_yes,crit2_middle_east_n_africa[1])
crit2_yes = np.append(crit2_yes,crit2_sub_africa[1])
crit3_yes = np.append(crit3_north_america[1], crit3_asia[1])
crit3_yes = np.append(crit3_yes,crit3_oceania[1])
crit3_yes = np.append(crit3_yes,crit3_europe[1])
crit3_yes = np.append(crit3_yes,crit3_south_america[1])
crit3_yes = np.append(crit3_yes,crit3_middle_east_n_africa[1])
crit3_yes = np.append(crit3_yes,crit3_sub_africa[1])

# Create array for no
crit1_no = np.append(crit1_north_america[0], crit1_asia[0])
crit1_no = np.append(crit1_no,crit1_oceania[0])
crit1_no = np.append(crit1_no,crit1_europe[0])
crit1_no = np.append(crit1_no,crit1_south_america[0])
crit1_no = np.append(crit1_no,crit1_middle_east_n_africa[0])
crit1_no = np.append(crit1_no,crit1_sub_africa[0])
crit2_no = np.append(crit2_north_america[0], crit2_asia[0])
crit2_no = np.append(crit2_no, 0)
crit2_no = np.append(crit2_no,crit2_europe[0])
crit2_no = np.append(crit2_yes,crit2_south_america[0])
crit2_no = np.append(crit2_no,crit2_middle_east_n_africa[0])
crit2_no = np.append(crit2_no,crit2_sub_africa[0])
crit3_no = np.append(crit3_north_america[0], crit3_asia[0])
crit3_no = np.append(crit3_no,crit3_oceania[0])
crit3_no = np.append(crit3_no,crit3_europe[0])
crit3_no = np.append(crit3_no,crit3_south_america[0])
crit3_no = np.append(crit3_no,crit3_middle_east_n_africa[0])
crit3_no = np.append(crit3_no,crit3_sub_africa[0])

# create total observations
total_north_america = sum(crit1_north_america)
total_asia = sum(crit1_asia)
total_oceania = sum(crit1_oceania)
total_europe = sum(crit1_europe)
total_south_america = sum(crit1_south_america)
total_middle_east_n_africa = sum(crit1_middle_east_n_africa)
total_sub_africa = sum(crit1_sub_africa)

total_obs_region = np.append(total_north_america, total_asia)
total_obs_region = np.append(total_obs_region, total_oceania)
total_obs_region = np.append(total_obs_region, total_europe)
total_obs_region = np.append(total_obs_region, total_south_america)
total_obs_region = np.append(total_obs_region, total_middle_east_n_africa)
total_obs_region = np.append(total_obs_region, total_sub_africa)




# Create bar chart with absolute values

trace0 = go.Bar(
    x= regions,
    y= total_obs_region,
    name = 'Observations in region'
)
trace1 = go.Bar(
    x= regions,
    y= crit1_yes,
    name = 'Political, economic, religious, or social'
)
trace2 = go.Bar(
    x= regions,
    y= crit2_yes,
    name = 'Coerce, intimidate, or publicize'
)
trace3 = go.Bar(
    x= regions,
    y= crit3_yes,
    name = 'Outside international humatarian law'
)

layout = go.Layout(
    title = 'Absolute accordance to the 3 criteriums',
    barmode='group'
)

data = [trace0, trace1, trace2, trace3]

figure = dict(data = data, layout = layout)
iplot(figure)




# create relative = crit 1 / total

rel_crit1_yes = crit1_yes / total_obs_region
rel_crit2_yes = crit2_yes / total_obs_region
rel_crit3_yes = crit3_yes / total_obs_region

trace1 = go.Bar(
    x= regions,
    y= rel_crit1_yes,
    name = 'Political, economic, religious, or social'
)
trace2 = go.Bar(
    x= regions,
    y= rel_crit2_yes,
    name = 'Coerce, intimidate, or publicize'
)
trace3 = go.Bar(
    x= regions,
    y= rel_crit3_yes,
    name = 'Outside international humatarian law'
)

layout = go.Layout(
    title = 'relative accordance to the 3 criteriums',
    barmode='group'
)

data = [trace1, trace2, trace3]

figure = dict(data = data, layout = layout)
iplot(figure)




# terrorist attacks in South America only
terror_SA = data_terrorism[(data_terrorism.region == 3)]

SA_countries = np.asarray(['BRA', 'COL', 'ARG', 'VEN', 'PER', 'CHI', 'ECU', 'BOL', 'PAR', 'URU', 'GUY', 'SUR', 'FAL', 'FRG'])
SA_population = np.asarray([205823665, 47220856, 43886748, 30912302, 30741062, 17650114, 16080778, 10969649, 6862812, 3351016, 735909, 585824, 2931, 231167])

#terrorist attacks per 100,000 people in country
terror_percountrySA = np.asarray(terror_SA.groupby('country').country.count())
terror_percapitaSA = np.round(terror_percountrySA / SA_population * 100000 , 2)

terror_scale = [[0, 'rgb(252, 232, 213)'], [1, 'rgb(240, 140, 45)']]

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = terror_scale,
        showscale = False,
        locations = SA_countries,
        locationmode = 'South America',
        z = terror_percapitaSA, 
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            )
        )]

layout = dict(
         title = 'Terrorist Attacks per 100,000 People in South America (1970-2015)',
         geo = dict(
             scope = 'south america',
             projection = dict(type = 'natural earth'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
         )

figure = dict(data = data, layout = layout)
iplot(figure)




count_year = data_terrorism.groupby(['iyear']).count()
death_year = data_terrorism.groupby(['iyear']).mean()

f1 = plt.figure()
ax1 = f1.add_subplot(211)
ax1.plot(count_year.index, count_year.nkill)
ax1.set(title='Total fatalities over time',xlabel='Year',ylabel='Fatalities')

f2 = plt.figure()
ax2 = f2.add_subplot(212)
ax2.plot(death_year.index, death_year.nkill)
ax2.set(title='Average fatalities per terrorist attack',xlabel='Year',ylabel='Fatalities')

plt.show()




# Attack type
data_terrorism['count'] = 1
by_year = (data_terrorism.groupby('iyear').agg({'count':'sum'}))
attack_type = data_terrorism.groupby('attacktype1')['count'].count().reset_index()
total = attack_type['count'].sum()
attack_type['Percentage'] = attack_type.apply(lambda x : (x['count']/total) * 100, axis=1)

plt.figure(figsize=[16,8])
sns.pointplot(x='attacktype1', y='Percentage', data=attack_type, color='yellow', rotation=30)
plt.xlabel('Attack Type', size=16)
plt.ylabel('Percentage', size=16)

data_matrix = [['Number', 'Attack_Type']
               ['1', 'Assassination'],
               ['2', 'Armed_Assault'],
               ['3', 'Bombing/Explosion'],
               ['4', 'Hijacking'],
               ['5', 'Hostage_Taking(barricade_Incident)'],
               ['6', 'Hostage_Taking(Kidnapping)'],
               ['7', 'Facility/Infrastructure_Attack'],
               ['8', 'Unarmed_Assault'],
               ['9', 'Unknown']]
               
table = ff.create_table(data_matrix)
py.iplot(table, filename='simple_table')




g = sns.factorplot(x="success", hue="attacktype1", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs success given attacktype1")




g = sns.factorplot(x="suicide", hue="success", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("number of attacks vs suicide given success")




g = sns.factorplot(x="targtype1", hue="success", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("number attacks vs targtype1 given success")




g = sns.factorplot(x="success", hue="property", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs success given property damage")




g = sns.factorplot(x="success", hue="propextent", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("nr attacks vs success given how big property damage")




g = sns.factorplot(x="success", hue="ishostkid", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs succes given victims taken hostage/kidnapped")




g = sns.factorplot(x="success", hue="ransom", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs succes given ransom")




g = sns.factorplot(x="success", hue="hostkidoutcome", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs succes given outcome hostages/kidnappes")




g = sns.factorplot(x="success", hue="INT_IDEO", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs succes given the ideology comparison")




g = sns.factorplot(x="success", hue="INT_MISC", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs succes given the location/nationality comparison (miscellaneous)")




g = sns.factorplot(x="success", hue="INT_LOG", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs succes given the location/nationality comparison (logictically)")




g = sns.factorplot(x="weaptype1", hue="success", kind="count",
                   data=data_terrorism,size=6,palette="muted")
g.set_ylabels("Number of attacks vs weapon type given the success")




# Create subarrays for the different regions
data_north_america = data_terrorism[(data_terrorism.region == 1) |  (data_terrorism.region == 2)]
data_asia = data_terrorism[(data_terrorism.region == 4) | (data_terrorism.region == 5) | (data_terrorism.region == 6) | (data_terrorism.region == 7)]
data_oceania = data_terrorism[(data_terrorism.region == 12)]
data_europe = data_terrorism[(data_terrorism.region == 8) | (data_terrorism.region == 9)]
data_south_america = data_terrorism[(data_terrorism.region == 3)]
data_middle_east_n_africa = data_terrorism[(data_terrorism.region == 10)]
data_sub_africa = data_terrorism[(data_terrorism.region == 11)]

# create x value: nr of attacks per country
attacks_north_america = np.asarray(data_north_america.groupby('country').country.count())
attacks_asia = np.asarray(data_asia.groupby('country').country.count())
attacks_oceania = np.asarray(data_oceania.groupby('country').country.count())
attacks_europe = np.asarray(data_europe.groupby('country').country.count())
attacks_south_america = np.asarray(data_south_america.groupby('country').country.count())
attacks_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('country').country.count())
attacks_sub_africa = np.asarray(data_sub_africa.groupby('country').country.count())

# create y value: succesfullness: nr. of kills, nr. of wounded, nr. of propoertydamage
nkill_north_america = np.asarray(data_north_america.groupby('country').nkill.sum())
nkill_asia = np.asarray(data_asia.groupby('country').nkill.sum())
nkill_oceania = np.asarray(data_oceania.groupby('country').nkill.sum())
nkill_europe = np.asarray(data_europe.groupby('country').nkill.sum())
nkill_south_america = np.asarray(data_south_america.groupby('country').nkill.sum())
nkill_middle_east_n_africa = np.asarray(data_middle_east_n_africa.groupby('country').nkill.sum())
nkill_sub_africa = np.asarray(data_sub_africa.groupby('country').nkill.sum())




# Plot scatter plot (titel klopt nog niet)
trace = go.Scatter(
    x = attacks_south_america,
    y = nkill_south_america,
    mode = 'markers'
)

layout = dict(title = 'Styled Scatter',
              yaxis = dict(zeroline = True),
              xaxis = dict(zeroline = True)
             )

# Plot and embed in ipython notebook!
figure = dict(data = [trace], layout = layout)
iplot(figure)




# insert data about world population (average of a country between 1970 and 201)

names_country = np.array(['Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya', 'Madagascar', 'Malawi', 
                          'Mauritius', 'Mozambique', 'Rwanda', 'Seychelles', 'Somalia', 'South Sudan', 'Uganda', 
                          'Tanzania', 'Zambia', 'Zimbabwe', 'Angola', 'Cameroon', 'Central African Republic', 'Chad', 
                          'Democratic Republic of the Congo', 'Equatorial Guinea', 'Gabon', 'Algeria', 'Egypt', 
                          'Libya', 'Morocco', 'Sudan', 'Tunisia', 'Western Sahara', 'Botswana', 'Lesotho', 'Namibia', 
                          'South Africa', 'Swaziland', 'Benin', 'Burkina Faso', 'Gambia', 'Ghana', 'Guinea', 
                          'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 
                          'Togo', 'China', 'Hong Kong', 'Taiwan', 'North Korea', 'Japan', 'South Korea', 'Kazakhstan', 'Kyrgyzstan', 
                          'Tajikistan', 'Turkmenistan', 'Uzbekistan', 'Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Iran', 'Maldives', 
                          'Nepal', 'Pakistan', 'Sri Lanka', 'Brunei', 'Cambodia', 'Indonesia', 'Malaysia', 'Myanmar', 'Philippines', 
                          'Singapore', 'Thailand', 'Armenia', 'Azerbaijan', 'Bahrain', 'Cyprus', 'Georgia', 'Iraq', 'Israel', 'Jordan', 
                          'Kuwait', 'Lebanon', 'Qatar', 'Saudi Arabia', 'Syria', 'Turkey', 'United Arab Emirates', 'Yemen', 'Belarus', 
                          'Bulgaria', 'Czech Republic', 'Hungary', 'Poland', 'Moldova', 'Romania', 'Russia', 'Slovak Republic', 'Ukraine', 
                          'Denmark', 'Estonia', 'Finland', 'Iceland', 'Ireland', 'Latvia', 'Lithuania', 'Norway', 'Sweden', 'United Kingdom', 
                          'Albania', 'Andorra', 'Croatia', 'Greece', 'Italy', 'Malta', 'Montenegro', 'Portugal', 'Serbia', 'Slovenia', 'Spain', 
                          'Macedonia', 'Austria', 'Belgium', 'France', 'Germany', 'Luxembourg', 'Netherlands', 'Switzerland', 
                          'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Cuba', 'Dominica', 'Dominican Republic', 'Grenada', 
                          'Guadeloupe', 'Haiti', 'Jamaica', 'Martinique', 'St. Kitts and Nevis', 'St. Lucia', 'Trinidad and Tobago',
                          'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Mexico', 'Nicaragua', 'Panama', 'Argentina', 
                          'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Falkland Islands', 'French Guiana', 'Guyana', 'Paraguay', 
                          'Peru', 'Suriname', 'Uruguay', 'Venezuela', 'Canada', 'United States', 'Australia', 'New Zealand', 'Fiji', 
                          'New Caledonia', 'Papua New Guinea', 'Solomon Islands', 'Vanuatu', 'French Polynesia', 'Wallis and Futuna'])
 
avg_inhabitants = np.array([5929.10939130435, 461.982260869565, 568.661195652174, 3167.80902173913, 56660.6153478261, 26496.5380217391, 
                            13582.7968043478, 9863.82284782609, 1083.64836956522, 16245.7041956522, 7087.97323913044, 75.2548043478261, 
                            8030.06308695652, 6477.32982608696, 20852.8252608696, 29522.112326087, 9068.8527173913, 10345.934673913, 
                            14509.1303913043, 13196.8929347826, 3193.35939130435, 7325.28006521739, 41157.2444347826, 543.27952173913, 
                            1088.89243478261, 26797.3883043478, 61115.0571739131, 4508.86017391304, 25491.8166956522, 22802.6085217391, 
                            8361.28195652174, 268.547326086957, 1445.40223913043, 1638.97108695652, 1538.20356521739, 39382.7373695652, 
                            877.396826086957, 5890.58319565217, 10251.004326087, 1048.30447826087, 16443.0094782609, 7265.6877173913, 
                            1117.94847826087, 2553.27095652174, 10021.2752826087, 2348.29267391304, 9931.18865217391, 107011.648782609, 
                            8563.68869565217, 4493.72626086957, 4269.10167391304, 1163199.68545652, 5826.53819565218, 20129.2505652174, 
                            20663.6947391304, 122594.270434783, 43258.3279347826, 15532.5435869565, 4407.20291304348, 5482.9697826087, 
                            3862.84728260869, 21418.0874782609, 18216.3767391304, 112356.130934783, 531.21852173913, 921193.107956522, 
                            55578.0176956522, 243.8585, 20202.0932608696, 117190.784065217, 17287.3976304348, 277.343282608696, 10298.4438043478, 
                            187633.016391304, 19769.3171304348, 40896.353673913, 66855.7890652174, 3466.61245652174, 56205.7741304348, 
                            3069.41267391304, 7400.00323913044, 634.329739130435, 848.754391304348, 4804.98969565217, 20352.562, 5170.8377826087, 
                            4337.16436956522, 1951.00813043478, 3259.02689130435, 707.046130434783, 17364.4145434783, 13578.3613478261, 
                            56086.0281304348, 3142.34897826087, 14552.4723695652, 9718.86830434783, 8298.81045652174, 10303.6185869565, 
                            10318.2473913043, 37057.3202608696, 4111.80986956522, 21981.0171086957, 142224.572043478, 5193.58193478261, 
                            48784.3528913043, 5257.32304347826, 1425.63054347826, 5043.43595652174, 263.84202173913, 3770.52060869565, 
                            2406.43467391304, 3396.71195652174, 4381.9037826087, 8724.50784782609, 58607.5434782609, 2897.0577826087, 
                            56.5975217391304, 4516.65902173913, 10381.4900652174, 57283.226826087, 371.455673913043, 600.830913043478, 
                            10020.9863478261, 9132.0465, 1934.8962173913, 40429.5194130435, 1971.92363043478, 7937.46189130435, 10223.325673913, 
                            57694.2000434783, 79944.3875, 414.953630434783, 15191.1485, 6945.57619565217, 78.5750652173913, 270.924413043478, 
                            262.701282608696, 10518.807, 71.9289782608695, 7526.48147826087, 98.6522173913043, 389.146391304348, 7519.14154347826, 
                            2452.75043478261, 361.613913043478, 45.4052826086957, 141.484826086957, 1198.13836956522, 214.779152173913, 
                            3319.18826086957, 5281.94815217391, 10224.1756304348, 5526.89834782609, 89266.8574565218, 4320.28106521739, 
                            2656.76630434783, 33709.3748913043, 7361.63471739131, 153928.587804348, 13721.1711956522, 35549.3115434783, 
                            10898.9652391304, 2.35376086956522, 136.069326086956, 754.081152173913, 4498.26436956522, 22577.5376521739, 
                            434.704, 3136.78836956522, 21106.998673913, 28502.9505217391, 262097.535130435, 17685.9465652174, 3629.91191304348, 
                            735.87102173913, 186.23547826087, 4826.96276086956, 350.2555, 163.026043478261, 202.665760869565, 12.716347826087])    

inhabitants_data = pd.DataFrame({"country":names_country,"inhabitants":avg_inhabitants})
inhabitants_data.head()




# merge inhabitants with grouped data

# Create nr. of attacks per country for the world
def create_df_grouped(data):
    dfout = pd.DataFrame({'country_txt':data['country_txt'].unique() ,
                         'country': len(data['country']) })
    return dfout

attacks_world_grouped = data_terrorism.groupby('country_txt').apply(create_df_grouped)

# Describe terrorism database
attacks_world_grouped

# to test
# data['country_txt'].groupby(data['country_txt']).describe()




merged_data = attacks_world_grouped.set_index('country_txt').join(inhabitants_data.set_index('country'))

merged_data['nattacks_inhabitants'] = merged_data.country / merged_data.inhabitants 
merged_data['nkill_inhabitants'] = 
merged_data




# create array with number of kills per country and merge it with the dataset above. 
# then create the subarrays done above and insert into scatter plot




##graph 1
# create array: number of attacks per attacktype1
nr_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').attacktype1.count())

# create array: sum of persons killed per attacktype1
nkill_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').nkill.sum())
# Average of persons killed per attack (per attacktype1)
average_nkill = np.divide(nkill_attacktype1, nr_attacktype1) 

# create array with names of attacktypes
attacktype1_names = np.array(['Assassination','Armed Attack','Bombing/Explosion','Hijacking','Hostage Taking barricade incident','Hostage Taking kidnapping','Facility/Infrastructure Attack','Unarmed Assault','Unknown'])
print(attacktype1_names)

##graph 2
# create number of deaths over total number of deaths per attacktype
total_deaths = sum(nkill_attacktype1)
average_nkill2 = np.divide(nkill_attacktype1, total_deaths) 
average_nkill_kills = average_nkill2*100

##graph 3
# create array: sum of persons wounded per attacktype1
nwound_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').nwound.sum())
# Average of persons wounded per attack (per attacktype1)
average_nwound = np.divide(nwound_attacktype1, nr_attacktype1) 

## graph 4
# create number of wounded over total number of wounded per attacktype
total_wounded = sum(nwound_attacktype1)
average_nwound2 = np.divide(nwound_attacktype1, total_wounded) 
average_nwound_wounded = average_nwound2*100

## graph 5
# create array: sum of propextent per attacktype1
propextent_attacktype1 = np.asarray(data_terrorism.groupby('attacktype1').propextent.sum())
# Average of property damage per attack (per attacktype1)
average_propextent = np.divide(propextent_attacktype1, nr_attacktype1) 

## graph 6
# create extent of property damage over total number of wounded per attacktype
total_property = sum(propextent_attacktype1)
average_propextent2 = np.divide(propextent_attacktype1, total_property) 
average_propextent_propextent = average_propextent2*100





# Create dataframe
attacktype_data = pd.DataFrame({"attacktype1_names":attacktype1_names,"nr_attacktype1":nr_attacktype1,"nkill_attacktype1":nkill_attacktype1,"average_nkill":average_nkill, "average_nkill_kills":average_nkill_kills,"average_nwound":average_nwound,"average_nwound_wounded":average_nwound_wounded,"average_propextent":average_propextent,"average_propextent_propextent":average_propextent_propextent})
attacktype_data.head()





#sort the dataframe from large to small
sorted_attacktype_data = attacktype_data.sort_values(by='average_nkill', ascending=0)

#sort the dataframe from large to small
sorted1_attacktype_data = attacktype_data.sort_values(by='average_nkill_kills', ascending=0)

#sort the dataframe from large to small
sorted2_attacktype_data = attacktype_data.sort_values(by='average_nwound', ascending=0)

#sort the dataframe from large to small
sorted3_attacktype_data = attacktype_data.sort_values(by='average_nwound_wounded', ascending=0)

#sort the dataframe from large to small
sorted4_attacktype_data = attacktype_data.sort_values(by='average_propextent', ascending=0)

#sort the dataframe from large to small
sorted5_attacktype_data = attacktype_data.sort_values(by='average_propextent_propextent', ascending=0)




##make barplot
ax = sns.barplot(y='attacktype1_names',x='average_nkill', data=sorted_attacktype_data, color="#00035b", palette="Reds_r")
#set x and y label
ax.set_xlabel("Average number deaths per attack", size=10, alpha=1)
ax.set_ylabel("Attacktype Names", size=10, alpha=1)
#set limit to y axis
ax.set(xlim=(0, 10))
#set title
ax.set_title("The average number of deaths per attack given the attack type", fontsize=12, alpha=1)
#set size and color parameters
ax.tick_params(labelsize=10,labelcolor="black")
sns.plt.show()




##make barplot: moet eigenlijk in de vorm van een pie chart 
ax = sns.barplot(y='attacktype1_names',x='average_nkill_kills', data=sorted1_attacktype_data, color="#00035b", palette="Reds_r")
#set x and y label
ax.set_xlabel("Mortality rate(?) per attacktype", size=10, alpha=1)
ax.set_ylabel("Attacktype names", size=10, alpha=1)
#set limit to y axis
ax.set(xlim=(0, 100))
#set title
ax.set_title("Deaths per attack type compared to total number of killed people (in %)", fontsize=12, alpha=1)
#set size and color parameters
ax.tick_params(labelsize=10,labelcolor="black")
sns.plt.show()




trace_attacktype_nkill = go.Pie(labels=attacktype1_names, values=average_nkill_kills)

iplot([trace_attacktype_nkill], filename="attacktype1_nkill_pie")




##make barplot
ax = sns.barplot(y='attacktype1_names',x='average_nwound', data=sorted2_attacktype_data, color="#00035b", palette="Blues_r")
#set x and y label
ax.set_xlabel("Average number of wounded people per attack", size=10, alpha=1)
ax.set_ylabel("Attacktype names", size=10, alpha=1)
#set limit to y axis
ax.set(xlim=(0, 30))
#set title
ax.set_title("The average number of wounded people per attack given the attack type", fontsize=12, alpha=1)
#set size and color parameters
ax.tick_params(labelsize=10,labelcolor="black")
sns.plt.show()




##make barplot
ax = sns.barplot(y='attacktype1_names',x='average_nwound_wounded', data=sorted3_attacktype_data, color="#00035b", palette="Blues_r")
#set x and y label
ax.set_xlabel("Percentage of wounded people per attack type", size=10, alpha=1)
ax.set_ylabel("Attacktype names", size=10, alpha=1)
#set limit to y axis
ax.set(xlim=(0, 100))
#set title
ax.set_title("The percentage of wounded people per attack type, compared to total (in %)", fontsize=12, alpha=1)
#set size and color parameters
ax.tick_params(labelsize=10,labelcolor="black")
sns.plt.show()




trace_attacktype_nwound = go.Pie(labels=attacktype1_names, values=average_nwound_wounded)

iplot([trace_attacktype_nwound], filename="attacktype1_nwound_pie")




##make barplot
ax = sns.barplot(y='attacktype1_names',x='average_propextent', data=sorted4_attacktype_data, color="#00035b", palette="Greens_r")
#set x and y label
ax.set_xlabel("Average property damage per attack", size=10, alpha=1)
ax.set_ylabel("Attacktype names", size=10, alpha=1)
#set limit to y axis
ax.set(xlim=(0, 4))
#set title
ax.set_title("The average extent of property damage per attack type", fontsize=12, alpha=1)
#set size and color parameters
ax.tick_params(labelsize=10,labelcolor="black")
sns.plt.show()





##make barplot
ax = sns.barplot(y='attacktype1_names',x='average_propextent_propextent', data=sorted5_attacktype_data, color="#00035b", palette="Greens_r")
#set x and y label
ax.set_xlabel("Percentage of property damage compared to total", size=10, alpha=1)
ax.set_ylabel("Attacktype names", size=10, alpha=1)
#set limit to y axis
ax.set(xlim=(0, 100))
#set title
ax.set_title("Percentage of property damage per attack type compared to total damage (in %)", fontsize=12, alpha=1)
#set size and color parameters
ax.tick_params(labelsize=10,labelcolor="black")
sns.plt.show()




trace_attacktype_propextent = go.Pie(labels=attacktype1_names, values=average_propextent_propextent)

iplot([trace_attacktype_propextent], filename="attacktype1_propextent_pie")




fig = {
  "data": [
    {
      "values": average_nkill_kills,
      "labels": attacktype1_names
        ,
    "text":"Property Damage",
      "textposition":"inside",
      "domain": {"x": [0, .30]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    },     
      {
      "values": average_nwound_wounded,
      "labels": attacktype1_names
        ,
    "text":"nkill",
      "textposition":"inside",
      "domain": {"x": [.35, .65]},
      "name": "",
      "hoverinfo":"label+percent+name",
          "hole": .4,
      "type": "pie"
    },     
    {
      "values": average_propextent_propextent,
      "labels":  attacktype1_names
        ,
      "text":"Nwound",
      "textposition":"inside",
      "domain": {"x": [.70, 1]},
      "name": "",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    }],
  "layout": {
        "title":"Share of kills, wounded and property damage per attack type", "showlegend":False,
        "annotations": [
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Killed",
                "x": 0.13,
                "y": 0.5
            },
             {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Wounded",
                "x": 0.50,
                "y": 0.5
            },
            {
                "font": {
                    "size": 10
                },
                "showarrow": False,
                "text": "Property",
                "x": 0.885,
                "y": 0.5
            }
        ]
    }
}
iplot(fig, filename='donut')






