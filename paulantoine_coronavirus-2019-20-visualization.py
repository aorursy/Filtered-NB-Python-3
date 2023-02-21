#!/usr/bin/env python
# coding: utf-8



# Importing packages
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import gc

pd.plotting.register_matplotlib_converters()

# matplotlib config

plt.style.use('fivethirtyeight') #'seaborn-notebook'
plt.rcParams.update({'font.size': 14})

# Load data
raw_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv',
                parse_dates = ['ObservationDate','Last Update'],
                dtype={'Confirmed':np.int32, 'Deaths':np.int32, 'Recovered':np.int32})

country_profile_variables = pd.read_csv('../input/undata-country-profiles/country_profile_variables.csv')

print(f"donnée la plus récente :{raw_data['Last Update'].max()}, {raw_data['ObservationDate'].max()}")




# prepare data

# country population :
# normalize country names, add missing country population data (cf data[data.isnull().any(axis=1)])

country_population = country_profile_variables[['country', 'Population in thousands (2017)']].copy()

country_population.columns = ["Country", "population"]
country_name_correction = {
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Brunei Darussalam': 'Brunei',
    'Czechia':'Czech Republic',
    'China, Hong Kong SAR':'Hong Kong',
    'Democratic Republic of the Congo': 'Congo (Kinshasa)',
    'Congo': 'Congo (Brazzaville)',
    'Iran (Islamic Republic of)':'Iran',
    'China, Macao SAR':'Macau',
    'China':'Mainland China',
    'The former Yugoslav Republic of Macedonia':'North Macedonia',
    'Republic of Moldova': 'Moldova',
    'Russian Federation':'Russia',
    'Republic of Korea':'South Korea',
    'State of Palestine': 'Palestine',
    'United Kingdom':'UK',
    'United Republic of Tanzania': 'Tanzania',
    'United States of America':'US',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Viet Nam':'Vietnam'
    } #   'North Ireland',          'Saint Barthelemy',

def normalize_country_name(name):
    if name in country_name_correction:
        return country_name_correction[name]
    else:
        return name
country_population["Country"] = country_population.Country.apply(normalize_country_name)

country_population = country_population.append(pd.DataFrame([
    ['Taiwan',23780],
    ['Diamond Princess',4],
    ['Guernsey', 67],
    ['Jersey', 98],
    ['Kosovo', 1831],
    ['Vatican City', 1],
    ['Reunion', 860],
    ['Ivory Coast', 2490]], columns=country_population.columns), ignore_index=True)

raw_data['Country/Region'] = raw_data['Country/Region'].apply(str.strip)
raw_data.fillna(value='NA', inplace=True)

# deduplicate to be safe

data = raw_data.groupby(['Country/Region','Province/State','ObservationDate'])[['Confirmed', 'Deaths', 'Recovered']].max().reset_index()
data.rename(columns={'Country/Region': 'Country'}, inplace=True)

data.loc[(data.Country == 'Others'), 'Country'] = 'Diamond Princess'
data.loc[(data.Country == 'Republic of the Congo'), 'Country'] = 'Congo (Kinshasa)'
data.loc[(data.Country == 'The Gambia') | (data.Country == 'Gambia, The'), 'Country'] = 'Gambia'
data.loc[(data.Country == 'The Bahamas')| (data.Country == 'Bahamas, The'), 'Country'] = 'Bahamas'
data.loc[(data.Country == 'occupied Palestinian territory'), 'Country'] = 'Palestine'
data.loc[(data.Country == 'Eswatini'), 'Country'] = 'Swaziland'
data.loc[(data.Country == 'Republic of Ireland'), 'Country'] = 'Ireland'

data = data.merge(country_population, on='Country', how="left")

# split Hubei from China
data.loc[data['Province/State'] == 'Hubei', 'population'] = 58500
data.loc[data['Country']        == 'China', 'Country'] = 'China other'
data.loc[data['Province/State'] == 'Hubei', 'Country'] = 'Hubei'


# segment column is either top country or rest of the world

def get_place(row):
    """get_place return top country or 'Rest of the World'"""
    # compute get_place.top_countries only once
    if not hasattr(get_place, "top_countries"):
        get_place.top_countries = set(data
                    .groupby(['Country', 'ObservationDate'])['Confirmed'].sum().reset_index()\
                    .groupby('Country').max().reset_index()\
                    .sort_values('Confirmed', ascending=False)['Country'].head(15)
                    )
        
    if row['Country'] in get_place.top_countries:
        return row['Country']
    else: return 'Rest of the World'
    

daily = data.groupby(['Country', 'ObservationDate']).agg({
            'Confirmed':'sum', 
            'Deaths':'sum',
            'Recovered':'sum',
            'population':'mean',
}).reset_index()
daily = daily[daily.Confirmed > 0]
daily.rename(columns={'ObservationDate': 'Date'}, inplace=True)
daily['segment'] = daily.apply(lambda row: get_place(row), axis=1)
daily['confirmed_per_population'] = daily.Confirmed / daily.population
daily['deaths_per_population'] = daily.Deaths / daily.population
daily.sort_values(['Country', 'Date'], inplace=True)
daily['new_confirmed'] = daily.groupby("Country")["Confirmed"].diff().fillna(0)
daily['composite'] = 40 * daily.Deaths + daily.Confirmed


#ratio score / population
df = daily.groupby(['segment']).population.mean().reset_index()
df.loc[df.segment == "Rest of the World", "population"] =     7530000 - df.loc[(df.segment != "Rest of the World") , 'population'].sum()
df.columns = ['segment', 'segment_population']
daily = daily.merge(df, on='segment')

# latest data

latest = daily[daily.Date == daily.Date.max()]

#data.head()
daily.head()




df = daily.groupby(['segment']).population.mean().reset_index()
df.loc[df.segment == "Rest of the World", "population"] =     7530000 - df.loc[(df.segment != "Rest of the World") , 'population'].sum()




print ('Total confirmed cases: %.d' %np.sum(latest['Confirmed']))
print ('Total death cases: %.d' %np.sum(latest['Deaths']))
print ('Total recovered cases: %.d' %np.sum(latest['Recovered']))

df = latest.groupby('segment').agg({
            'Confirmed':'sum', 
            'Deaths':'sum',
            'Recovered':'sum',
            'population':'min',
            #'ratio':'mean'
})
df['Death Rate'] = df['Deaths'] / df['Confirmed'] * 100
df['Recovery Rate'] = df['Recovered'] / df['Confirmed'] * 100
df




# Confirmed Cases World ex-China
worldstat = latest[(latest.Country != 'Mainland China') & (latest.Country != 'Hubei')].groupby('Country').sum()
df = worldstat.sort_values('Confirmed', ascending=False).head(20)
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
sns.barplot(df.Confirmed, df.index)
plt.title('Confirmed cases World ex-China (top 20)')
plt.yticks(fontsize=20)

# Death Cases World ex-China
df = worldstat.sort_values('Deaths', ascending=False).head(20)
#df = df[df.Deaths>0]
plt.subplot(1, 2, 2)
sns.barplot(df.Deaths, df.index)
plt.title('Deaths cases World ex-China (top 20)')
plt.yticks(fontsize=16)

plt.show()
df




# ratio confirmed / population

plt.figure(figsize=(20,18))
worldstat = latest[latest.population > 100].groupby('Country').sum()

df = worldstat.sort_values('confirmed_per_population', ascending=False)
df = df.iloc[:50, :]
plt.subplot(1, 2, 1)
sns.barplot(df.confirmed_per_population, df.index)
plt.title('confirmed / population 000 (top 50)')
plt.yticks(fontsize=16)


df = worldstat.sort_values('deaths_per_population', ascending=False)
df = df.iloc[:50, :]
plt.subplot(1, 2, 2)
sns.barplot(df.deaths_per_population, df.index)
plt.title('deaths / population 000 (top 50)')
plt.yticks(fontsize=16)
plt.show()




# composite score

df = pd.pivot_table(daily.dropna(subset=['composite']),
                    index='Date', 
                    columns='segment',
                    values='composite',
                    aggfunc=np.sum)\
        .fillna(method='ffill')
df = df.reindex(df.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,10))

n=df.shape[1] + 1
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(df, marker='o')
plt.yscale('log')
plt.title('composite score (Confirmed + 20 * Death)',fontsize=20)
plt.legend(df.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()
df.head()




# composite score / population
df = daily[daily.segment != 'Diamond Princess'][['Date', 'segment', 'composite', 'segment_population']]
df['composite_per_segment_population'] = df.composite / df.segment_population

df = pd.pivot_table(
                df.dropna(subset=['composite_per_segment_population']),
                index='Date', 
                columns='segment',
                values='composite_per_segment_population',
                aggfunc=np.sum)\
    .fillna(method='ffill')

df =df.reindex(df.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,10))

n=df.shape[1] + 0
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(df, marker='o')
plt.yscale('log')
plt.title('composite score / population 000', fontsize=20)
plt.legend(df.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




# composite score / population
df = daily[(daily.segment != 'Diamond Princess') &
           (daily.segment != 'Hubei') &
           (daily.segment != 'Iran') &
           (daily.segment != 'Mainland China') &
           (daily.segment != 'Rest of the World')][['Date', 'segment', 'composite', 'segment_population']]
df['composite_per_segment_population'] = df.composite / df.segment_population

confirm = pd.pivot_table(
                df.dropna(subset=['composite_per_segment_population']),
                index='Date', 
                columns='segment',
                values='composite_per_segment_population',
                aggfunc=np.sum)\
    .fillna(method='ffill')

confirm =confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,12))

n=confirm.shape[1] + 1
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
plt.yscale('log')
plt.title('composite score / population 000 (excluding China, Iran)', fontsize=20)
plt.legend(confirm.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




# composite score / population
df = daily[(daily.segment != 'Diamond Princess') &
           (daily.segment != 'Hubei') &
           (daily.segment != 'Iran') &
           (daily.segment != 'Mainland China') &
           (daily.segment != 'Rest of the World') &
           (daily.Date >= '2020-03-01')][['Date', 'segment', 'composite', 'segment_population']]
df['composite_per_segment_population'] = df.composite / df.segment_population

confirm = pd.pivot_table(
                df.dropna(subset=['composite_per_segment_population']),
                index='Date', 
                columns='segment',
                values='composite_per_segment_population',
                aggfunc=np.sum)\
    .fillna(method='ffill')

confirm =confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,12))

n=confirm.shape[1] + 1
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
plt.yscale('log')
plt.title('composite score / population 000 (excluding China, Iran)', fontsize=20)
plt.legend(confirm.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




# evolution since 2020-03-13 vs +20% daily
start_date = "2020-03-13"

df = daily[(daily.segment != 'Diamond Princess') &
           (daily.segment !='Iran') & 
           (daily.segment != 'Hubei') &
           (daily.segment != 'Mainland China') &
           (daily.segment != 'Rest of the World') &
           (daily.segment != 'Turkey') &
           (daily.Date >= start_date)
          ].groupby(['segment', 'Date']).agg({
            'composite':'sum', 
            'population':'mean',
}).reset_index()

min_by_segment = df.groupby("segment").composite.min().to_dict()

df["min_by_segment"] = df.segment.apply(lambda x: min_by_segment[x])

df['ratio'] = df.composite     /df.min_by_segment    /np.exp(
        (df.Date - datetime.datetime.strptime(start_date, '%Y-%m-%d'))
        .apply(lambda x: x.total_seconds() 
        / 3600.0 / 120.0))

confirm = pd.pivot_table(df.dropna(subset=['ratio']), index='Date', 
                         columns='segment', values='ratio', aggfunc=np.sum).fillna(method='ffill')

confirm = confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,12))

n=confirm.shape[1] + 1
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
#plt.yscale('log')
plt.title('composite score (20 * deaths + confirmed) \nvs +20% daily\n compared to 2020-03-10', fontsize=20)
plt.legend(confirm.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




# evolution since 2020-03-10 vs +20% daily
start_date = "2020-03-14"

df = daily[(daily.segment != 'Diamond Princess') &
           (daily.segment !='Iran') & 
           (daily.segment != 'Hubei') &
           (daily.segment != 'Mainland China') &
           (daily.segment != 'Rest of the World') &
           (daily.Date >= start_date)
          ].groupby(['segment', 'Date']).agg({
            'Deaths':'sum', 
            'population':'mean',
}).reset_index()

min_by_segment = df.groupby("segment").Deaths.min().to_dict()

df["min_by_segment"] = df.segment.apply(lambda x: min_by_segment[x])

df['ratio'] = df.Deaths     /df.min_by_segment    /np.exp(
        (df.Date - datetime.datetime.strptime(start_date, '%Y-%m-%d'))
        .apply(lambda x: x.total_seconds() 
        / 3600.0 / 120.0))

confirm = pd.pivot_table(df.dropna(subset=['ratio']), index='Date', 
                         columns='segment', values='ratio', aggfunc=np.sum).fillna(method='ffill')

confirm = confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,12))

n=confirm.shape[1] + 1
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
#plt.yscale('log')
plt.title(f'deaths \nvs +20% daily\n compared to {start_date}', fontsize=20)
plt.legend(confirm.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




daily.head()




# confirmed

confirm = pd.pivot_table(daily.dropna(subset=['Confirmed']), index='Date', 
                         columns='segment', values='Confirmed', aggfunc=np.sum).fillna(method='ffill')

confirm = confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,10))

n=confirm.shape[1] + 2
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
plt.yscale('log')
plt.title('Confirmed Cases',fontsize=20)
plt.legend(confirm.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




# confirmed

temp = daily.groupby(['Date', 'segment'])['Confirmed'].sum()
temp = temp.reset_index()
temp = temp.assign(segment_max=temp.groupby('segment')['Confirmed'].transform('max'))
temp = temp.sort_values(by=['Date', 'segment_max'])

#temp = daily.assign(segment_max=daily.groupby('segment')['Confirmed'].transform('max'))
top = temp.groupby('segment')['Confirmed'].max().reset_index().sort_values('Confirmed', ascending=False)['segment'][:12].to_list()

plt.style.use('seaborn')
#sns.set(font_scale=1.1)
sns.set()

g = sns.FacetGrid(temp, col="segment", hue="segment", 
                  sharey=True, col_wrap=6, col_order =top )
g = g.map(plt.plot, "Date", "Confirmed").set(yscale = 'log')
#grid.set(yticks = ticks, yticklabels = labels)
#fig, ax = plt.subplots()
#ax.legend(fontsize=8)
g.set_titles(size=16)
g.set_xticklabels(rotation=90, fontsize=8)
plt.show()
temp




# confirmed / population
df = daily.groupby(['segment', 'Date']).agg({
            'Confirmed':'sum', 
            'population':'mean',
}).reset_index()
df.loc[df.segment == "Rest of the World", "population"] =     7530000 - df.loc[(df.segment != "Rest of the World") & (df.Date == df.Date.max()), 'population'].sum()
df['ratio'] = df.Confirmed / df.population

confirm = pd.pivot_table(df.dropna(subset=['ratio']), index='Date', 
                         columns='segment', values='ratio', aggfunc=np.sum).fillna(method='ffill')

confirm = confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,10))

n=confirm.shape[1] + 2
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
plt.yscale('log')
plt.title('Confirmed Cases / population 000', fontsize=20)
plt.legend(confirm.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




#France vs Italy confirmed
decalage = -8
confirm = pd.pivot_table(daily.dropna(subset=['Confirmed']), index='Date', 
                         columns='Country', values='Confirmed', aggfunc=np.sum).fillna(method='ffill')
confirm = confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

confirm = confirm[["Italy", "France"]]
confirm.loc[:,"France"] = confirm["France"].shift(decalage)

plt.figure(figsize=(20,15))

n=confirm.shape[1]
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
plt.yscale('log')
plt.title(f'Confirmed, France vs Italy shifted ({decalage})', fontsize=20) #,fontweight="bold"
plt.legend(confirm.columns, loc=2, fontsize=13)
plt.xticks(rotation=75)
plt.show()




#France vs Italy deaths
decalage = -10
confirm = pd.pivot_table(daily[daily.Date >= "2020-02-25"].dropna(subset=['Deaths']),
                         index='Date', 
                         columns='Country',
                         values='Deaths',
                         aggfunc=np.sum).fillna(method='ffill')
confirm = confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

confirm = confirm[["Italy", "France"]]
confirm.loc[:,"France"] = confirm["France"].shift(decalage)

plt.figure(figsize=(15,10))

n=confirm.shape[1]
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
#plt.yscale('log')
plt.title(f'Décès France vs Italy (décalage: {decalage})', fontsize=20) #,fontweight="bold"
plt.legend(confirm.columns, loc=2, fontsize=15)
plt.xticks(rotation=75)
plt.show()




#all coutries
confirm = pd.pivot_table(daily.dropna(subset=['Confirmed']), index='Date', 
                         columns='Country', values='Confirmed', aggfunc=np.sum).fillna(method='ffill')
confirm = confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

confirm = confirm.loc[:, confirm.loc[confirm.index.max(),:] > 1]

plt.figure(figsize=(20,30))

n=confirm.shape[1]
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
plt.yscale('log')
plt.title('Confirmed Cases (all countries with 2+)', fontsize=20) #,fontweight="bold"
plt.legend(confirm.columns, loc=2, fontsize=13)
plt.xticks(rotation=75)
plt.show()




# new cases
new_c = pd.pivot_table(daily.dropna(subset=['new_confirmed']), index='Date', 
                         columns='segment', values='new_confirmed', aggfunc=np.sum).fillna(method='ffill')

new_c = new_c.reindex(new_c.max().sort_values(ascending=False).index, axis=1)
#new_c=new_c[[ 'South Korea', 'Iran', 'Mainland China', 'Italy', 'France', 'Germany']]
plt.figure(figsize=(20,10))

n=new_c.shape[1] + 1
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(new_c,  marker='o', ms=10)
#plt.yscale('log')
plt.title('new cases excluding Hubei', fontsize=20)
plt.legend(new_c.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




#new cases
new_c = pd.pivot_table(daily.dropna(subset=['new_confirmed']), index='Date', 
                         columns='segment', values='new_confirmed', aggfunc=np.sum).fillna(method='ffill')

new_c = new_c.reindex(new_c.max().sort_values(ascending=False).index, axis=1)
#new_c=new_c[['Hubei', 'South Korea', 'Iran', 'Mainland China', 'Italy', 'France', 'Germany']]
plt.figure(figsize=(20,10))

n=new_c.shape[1] + 1
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(new_c, ls='', marker='o', ms=10)
plt.yscale('log')
plt.title('new cases (log scale)', fontsize=20)
plt.legend(new_c.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




#death

death = pd.pivot_table(daily.dropna(subset=['Deaths']), 
                         index='Date', columns='segment', values='Deaths', aggfunc=np.sum).fillna(method = 'ffill')
death = death.reindex(death.max().sort_values(ascending=False).index, axis=1)
death = death[death > 0]

plt.figure(figsize=(20,10))
plt.axes().set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, death.shape[1]+1)])

plt.plot(death, marker='o')
plt.yscale('log')
plt.title('Death Cases', fontsize=20)
plt.legend(death.columns, loc=2, fontsize=18)
plt.xticks(rotation=45)
plt.show()




#deaths confirmed / population
df = daily.groupby(['segment', 'Date']).agg({
            'Deaths':'sum', 
            'population':'mean',
}).reset_index()
df.loc[df.segment == "Rest of the World", "population"] =     7530000 - df.loc[(df.segment != "Rest of the World") & (df.Date == df.Date.max()), 'population'].sum()
df['ratio'] = df.Deaths / df.population
df = df[df['ratio'] > 0]

confirm = pd.pivot_table(df.dropna(subset=['ratio']), index='Date', 
                         columns='segment', values='ratio', aggfunc=np.sum).fillna(method='ffill')

confirm = confirm.reindex(confirm.max().sort_values(ascending=False).index, axis=1)

plt.figure(figsize=(20,10))

n=confirm.shape[1] + 2
ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.jet(i) for i in np.linspace(0, 1, n)])

plt.plot(confirm, marker='o')
plt.yscale('log')
plt.title('Deaths / population 000', fontsize=20)
plt.legend(confirm.columns, loc=2, fontsize=18)
plt.xticks(rotation=75)
plt.show()




population = 65000000
stabilisation = 0.6 # "immunité collective" théorique avec R0 = 2.5
taux_en_icu = 0.04
taux_sympto = 0.5
nb_patients_icu = population * stabilisation * taux_sympto * taux_en_icu
nb_lits_icu = 5000 * 2  # source https://www.euronews.com/2020/03/19/covid-19-how-many-intensive-care-beds-do-member-states-have
                        # suppose capacité doublée
nb_semaines_en_icu = 1.2 # médiane 8 jours
nb = nb_patients_icu * nb_semaines_en_icu / nb_lits_icu
print(f"nb de personnes en ICU {int(nb_patients_icu)}") 
print(f"durée pour passer la population sans que le système de santé soit débordé {int(nb)} semaines")

