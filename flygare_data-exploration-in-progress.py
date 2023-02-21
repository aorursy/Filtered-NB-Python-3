#!/usr/bin/env python
# coding: utf-8



from collections import OrderedDict
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.finance import candlestick2_ohlc
import seaborn as sns
from datetime import datetime 
from sklearn import linear_model
from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-pastel')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
elo = pd.DataFrame.from_csv("../input/fide_historical.csv")
elo = elo.reset_index()




elo['PubYear']     = elo.ranking_date.dt.year                      # Enriching the dataset PubYear
elo['PubMonth']    = elo.ranking_date.dt.month                     # Enriching the dataset PubMonth
elo['PubRunMonth'] = ((elo['PubYear']-2000)*12+elo['PubMonth'])-6  # Enriching the datasat PubRunMonth

# first period assumed to be three months
interval = [3.0]          + [round((d1-d0)/pd.Timedelta(1, unit='M'))
            for d0,d1 in zip(elo.ranking_date.unique()[0:-1],
                             elo.ranking_date.unique()[1:])]
elo['timedelta']   = elo['ranking_date'].map(dict(zip(elo.ranking_date.unique(),interval)))
elo['age']         = elo['PubYear']-elo['birth_year']
elo['listindex']   = elo['ranking_date'].map(dict(zip(elo.ranking_date.unique(),
                                                      range(0,elo.ranking_date.nunique()))))
for p in elo.name.unique():
    elo.loc[elo.name==p,'rating_move']        =-elo.loc[elo.name==p,'rating'].shift(1)        +elo.loc[elo.name==p,'rating']
elo['rating_move_timedelta']=elo['rating_move']/elo['timedelta']




# Manually reviewed and replaced
replacement = {
    'Bologan, Victor'             : 'Bologan, Viktor',
    'Bruzon Batista, Lazaro'      : 'Bruzon, Lazaro',
    'Dominguez Perez, Lenier'     : 'Dominguez, Leinier' ,
    'Dominguez Perez, Leinier'    : 'Dominguez, Leinier',
    'Dominguez, Lenier'           : 'Dominguez, Leinier',
    'Dreev, Aleksey'              : 'Dreev, Alexey',
    'Iturrizaga Bonelli, Eduardo' : 'Iturrizaga, Eduardo',
    'Kasparov, Gary'              : 'Kasparov, Garry',
    'Mamedyarov, Shakhriyaz'      : 'Mamedyarov, Shakhriyar',
    'McShane, Luke J'             : 'McShane, Luke J.',
    'Polgar, Judit (GM)'          : 'Polgar, Judit',
    'Sadler, Matthew D'           : 'Sadler, Matthew',
    'Short, Nigel D'              : 'Short, Nigel D.'}
for k,v in replacement.items():
    elo.loc[elo.name==k,'name'] = v




print('Day of month the data is released:\t%i' %elo.ranking_date.dt.day.unique()[0])
# always on the 27th; information can be discarded and regarded as a monthly dataset
NUMBER_DATAPOINTS_ = elo.ranking_date.nunique()
print('Number of rating list publications:\t%i' %NUMBER_DATAPOINTS_)
STARTDATE_=elo.ranking_date.min()
print('Date of first publication:\t\t%s'%STARTDATE_)
ENDDATE_=elo.ranking_date.max()
print('Date of first publication:\t\t%s'%ENDDATE_)




ax=elo.groupby('PubYear')['PubMonth'].nunique().plot(kind='bar')
ax.set_xlabel('Year')
ax.set_ylabel('Rating publications')
ax.set_title('Rating publications per year');
 




print(elo[elo.title!='g']         .groupby(['name'])         ['name', 'title'].max().values)




print(elo[elo.name=='Hou, Yifan']         .groupby(['name'])         [['name', 'title']].max().values) #'wg' counterexample




g=elo[elo['rank']==1][['ranking_date', 'name']]
clr={'Kasparov, Garry'   : 'b',
     'Topalov, Veselin'  : 'g',
     'Anand, Viswanathan': 'y',
     'Kramnik, Vladimir' : 'k',
     'Carlsen, Magnus'   : 'r'}




fig, ax = plt.subplots(figsize=(17,2))
ax.set_xlim(g.ranking_date.iloc[0],g.ranking_date.iloc[113])
for i in range(len(g)-1):
    ax=plt.barh(left=g.ranking_date.iloc[i],
            height=1,
            width=(g.ranking_date.iloc[i+1]-g.ranking_date.iloc[i]).days,
            bottom=0.5,
            edgecolor='none',
            color=clr[g.name.iloc[i]],
            label=g.name.iloc[i], alpha=0.5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.02, 1.0))
plt.yticks(visible=False)
plt.title('Number One player over time');




fig, ax = plt.subplots(figsize=(17,6))
for n in clr.keys():
    elo[elo['name']==n][['ranking_date','rating']]        .rename(columns={'rating': n})        .plot(ax=ax, label=n, color=clr[n])
plt.ylabel('Rating')
plt.xlabel('Date')
ax.legend(loc='lower right')
plt.title('Ratings over time of the number one players');




elo[elo.ranking_date==ENDDATE_].name
clr={'Carlsen, Magnus'       :'r',
     'So, Wesley'            :'b',
     'Kramnik, Vladimir'     :'k',
     'Caruana, Fabiano'      :'g',
     'Mamedyarov, Shakhriyar':'y'}




fig, ax = plt.subplots(figsize=(17,6))
names =clr.keys()
for n in names:
    elo[elo['name']==n][['ranking_date','rating']]        .rename(columns={'rating': n})        .plot(ax=ax, label=n, color=clr[n])
plt.ylabel('Rating')
plt.xlabel('Date')
ax.legend(loc='lower right')
plt.title('Ratings over time of the current top 5 players');




df=elo[elo['name']=='Carlsen, Magnus'][['ranking_date','rating']]
df['ryear'] = df.ranking_date.dt.year
df= df.groupby('ryear', as_index=True)  .agg({'rating': ['first','max', 'min','last']})  .loc[:,'rating']  .reset_index()
fig, ax = plt.subplots(1,1)
candlestick2_ohlc(ax, df['first'], df['max'], df['min'], df['last'], width=0.9)
ax.set_xticklabels(df.ryear)
ax.set_title('Candelstick chart per year (Magnus Carlsen)');




#Show six equal distribted dates from the observation period
dates = [list(elo.ranking_date.unique())[int(i)]
         for i in np.linspace(0,elo.ranking_date.nunique()-1,6).round()]

row, col = 3,2
fig, ax = plt.subplots(col,row, figsize=(17,6))
fig.tight_layout(h_pad=2.0)

for c in range(col):
    for r in range(row):
        d=dates[c*3+r]
        ax[c,r].set_xlim(2650,2900)
        ax[c,r].set_ylim(0,22)
        elo[elo.ranking_date==d].rating.hist(bins=20, edgecolor='w',ax=ax[c,r])
        ts = pd.to_datetime(str(d)) 
        ax[c,r].set_title(ts.strftime('%b-%Y'))
fig.subplots_adjust(top=0.90)        
fig.suptitle('Distrubtion of the Top100 over time');       




fig, ax = plt.subplots(3,2, figsize=(10,8), sharex=True)
fig.tight_layout()

top = (('Top10', 10, 'lightblue'),
       ('Top50', 50, 'slateblue'),
       ('Top100', 100, 'darkblue'))
for t, i in zip(top, range(len(top))):
    g=elo[elo['rank']<=t[1]]        .rename(columns={'rating': t[0]})        .groupby(['ranking_date'])        [t[0]].mean()        .to_frame()
    g['avg']=g[t[0]].rolling(window=10).mean()
    g['avgdelta']=g['avg']-g['avg'].shift(1)
    ax[i,0].plot(g[t[0]],color=t[2], label=t[0])
    ax[i,0].plot(g['avg'],color=t[2], ls='--', label='MA10')
    ax[i,1].plot(g['avgdelta'], label='1st deriv. MA10', color= t[2])
    ax[i,0].legend(loc='upper left')
    ax[i,1].legend(loc='upper left')
    ax[i,0].set_ylim(2630,2810)
    ax[i,1].set_ylim(-1,2.5)
    ax[i,1].axhline(0, c='k', lw=.5)
plt.xlabel('Date')
ax[0,0].set_title('Average rating over time');
ax[0,1].set_title('Changes in average rating over time');




top  = elo[['ranking_date', 'name','rating_move']]          .sort_values(['rating_move'], ascending=False)[:10]
flop = elo[['ranking_date', 'name', 'rating_move']]          .sort_values(['rating_move'])[:10]
g = pd.concat([top,flop]).sort_values(by='rating_move', ascending=False)

fig, ax = plt.subplots(2,1, figsize=(6,11), sharex=True)
fig.tight_layout()
rg=['r' if x<0 else 'g' for x in list(g.rating_move)]
ax[0].barh(np.arange(20),g.rating_move, label=g, color=rg)
ax[0].set_yticks(np.arange(20))
ax[0].set_yticklabels([str(n)+" ("+d.strftime('%b-%Y')+")" for n,d in zip(g.name, g.ranking_date)])
ax[0].set_title('Largest Rating Moves (per publishing period)');

top  = elo[['ranking_date', 'name','rating_move_timedelta']]          .sort_values(['rating_move_timedelta'], ascending=False)[:10]
flop = elo[['ranking_date', 'name', 'rating_move_timedelta']]          .sort_values(['rating_move_timedelta'])[:10]
g = pd.concat([top,flop]).sort_values(by='rating_move_timedelta', ascending=False)

rg=['r' if x<0 else 'g' for x in list(g.rating_move_timedelta)]
ax[1].barh(np.arange(20),g.rating_move_timedelta, label=g, color=rg)
ax[1].set_yticks(np.arange(20))
ax[1].set_yticklabels([str(n)+" ("+d.strftime('%b-%Y')+")" for n,d in zip(g.name, g.ranking_date)])
ax[1].set_title('Largest Rating Moves (per month)');




print('Unique number of player listed in dataset: %i' %elo.name.nunique())




fig, ax = plt.subplots(1,1, figsize=(17,5))
elo.groupby(['name'], as_index=False)    ['ranking_date'].count()    .groupby('ranking_date')    ['name'].count()    .reindex(np.linspace(1,NUMBER_DATAPOINTS_,NUMBER_DATAPOINTS_,dtype='int32'), fill_value=0 )    .plot(kind='bar')
ax.set_xlabel('Number of listing')
ax.set_ylabel('Number of players')
ax.set_xticks([0,1,2]+[i-1 for i in range(5,NUMBER_DATAPOINTS_,5)]+[112,113])
ax.set_xticklabels([1,2,3]+[i for i in range(5,NUMBER_DATAPOINTS_,5)]+[113,114])
ax.set_title('Frequency of a player listed');




## ToDo get the 18 players for Linear Regression




## ToDo add more stats here




tab=elo.groupby(['name'])['games','timedelta'].sum()
tab['AvgPlayedPerMonth'] = tab['games']/tab['timedelta']

fig, ax = plt.subplots(1,1, figsize=(17,5))
tab.sort_values('AvgPlayedPerMonth', ascending=False)    ['AvgPlayedPerMonth'][0:50]    .plot(kind='bar')
ax.set_title('Top 50 players games per month');




fig, ax = plt.subplots(1,1, figsize=(17,5))
tab.sort_values('timedelta', ascending=False)    ['AvgPlayedPerMonth'][0:50]    .plot(kind='bar')
ax.set_title('Games per month (filtered Top50 players by time top50');




g=elo.groupby('ranking_date').agg({'games':'sum','timedelta':'first', 'PubYear': 'first', 'PubMonth':'first'})
g['games_month']=g['games']/g['timedelta']/100
sns.regplot(y=g['games_month'], x=np.linspace(0,len(g['games_month']),len(g['games_month'])), order=1);
plt.ylabel('Games per month per player')
plt.xlim(-2,117)
plt.title('Games per month over time');
#ToDo Add labels




fig, ax = plt.subplots(1,1, figsize=(7,7))
ax = plt.subplot(projection='polar')
frac=1/12*2*np.pi
maxvalue=np.ceil(g.games_month.max())
color=cm.bone(np.linspace(0,1,25))#offset for darker color

for y,i in zip(range(2000,2018), range(0,18)):
    plt.plot((g[g.PubYear==y]['PubMonth'].values-1)*frac,
             g[g.PubYear==y]['games_month'].values,
             marker='o', lw=0, color=color[i], label=str(y))
ax.set_rmin(0)
ax.set_rmax(maxvalue)
ax.set_rticks([2.5,5,7.5,10])
ax.set_theta_offset(np.pi/2)
ax.set_thetagrids([30*i for i in range(12)])
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(bbox_to_anchor=(1.25, 1.0))
plt.title('Average games played by month');
### DEAD END ###
# I believe I am seeing not much here. The pattern is an artifact due the different publication intervals.




top = (('Top10', 10, 'lightblue'),
       ('Top50', 50, 'slateblue'),
       ('Top100', 100, 'darkblue'))

fig, ax = plt.subplots(1,1, figsize=(7,5))
for t in top:
    elo[elo['rank']<=t[1]]        .rename(columns={'age': t[0]})        .groupby(['ranking_date'])        [t[0]].mean()        .plot(color=t[2])
plt.legend(loc='lower left')
plt.xlabel('Date')
ax.set_title('Average age over time');




fig, ax = plt.subplots(3,2, figsize=(10,8), sharex=True)
fig.tight_layout()

top = (('Top10', 10, 'lightblue'),
       ('Top50', 50, 'slateblue'),
       ('Top100', 100, 'darkblue'))
for t, i in zip(top, range(len(top))):
    g=elo[elo['rank']<=t[1]]        .rename(columns={'age': t[0]})        .groupby(['ranking_date'])        [t[0]].mean()        .to_frame()
    g['avg']=g[t[0]].rolling(window=10).mean()
    g['avgdelta']=g['avg']-g['avg'].shift(1)
    ax[i,0].plot(g[t[0]],color=t[2], label=t[0])
    ax[i,0].plot(g['avg'],color=t[2], ls='--', label='MA10')
    ax[i,1].plot(g['avgdelta'], label='1st deriv. MA10', color= t[2])
    ax[i,0].legend(loc='upper left')
    ax[i,1].legend(loc='upper left')
    ax[i,0].set_ylim(28,34)
    ax[i,1].set_ylim(-0.5,0.5)
    ax[i,1].axhline(0, c='k', lw=.5)
plt.xlabel('Date')
ax[0,0].set_title('Average age over time');
ax[0,1].set_title('Changes in age over time');




yeargroups=((0,9), (9,12), (12,17))

fig, axs = plt.subplots(1,3, figsize=(17,5), sharey=True)
for ax, yg in zip(axs, yeargroups):
    for y in range(yg[0],yg[1]):
        date=STARTDATE_+pd.DateOffset(years=y)
        elo[elo.ranking_date==date]['age'].plot(kind='kde', label=str(date.year), ax=ax)
    ax.legend()
    ax.set_xlim(-1,75)
    ax.set_ylim(0,0.07)
    ax.set_xlabel('Average age')
fig.suptitle('Age distribution at start of the year');    




dates=[STARTDATE_+pd.DateOffset(years=y) for y in range(0,17)]
df=elo.loc[elo.ranking_date.isin(dates),['ranking_date','age']]
df["year"]=df.ranking_date.dt.year
           
pal = sns.cubehelix_palette(17, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="year", hue="year", aspect=8, size=0.9, palette=pal, xlim=(13,55),ylim=(0,0.07))
g.map(sns.kdeplot, "age", shade=True, alpha=1, lw=1.5)
g.map(sns.kdeplot, "age", color="w", lw=2) 
g.map(plt.axhline, y=0, lw=2, clip_on=False)

def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .25, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

g.map(label, "age")
g.fig.subplots_adjust(hspace=0,top=0.98)
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True);
g.fig.suptitle('Age distribution at start of the year (alternate plot)');




InOut = dict()

for i in range(0,NUMBER_DATAPOINTS_-1):
    name0 = set(elo[elo.listindex==i]['name'])
    name1 = set(elo[elo.listindex==i+1]['name'])
    a = elo[(elo.name.isin(name0-name1))&(elo.listindex==i)]        ['age'].mean()
    b = elo[(elo.name.isin(name1-name0))&(elo.listindex==i+1)]        ['age'].mean()
    InOut[i]=(a,b)

g=pd.DataFrame.from_dict(InOut).T
g['ranking_date']=list(elo.ranking_date.unique()[:-1])
g.columns = ['leaving', 'entering', 'ranking_date']   




fig, ax  = plt.subplots(1,1,figsize=(17, 6))
l=np.linspace(0,NUMBER_DATAPOINTS_-1,NUMBER_DATAPOINTS_-1)
sns.regplot(y=g.entering, x=l, color='green', label='entering')
sns.regplot(y=g.leaving, x=l, color='red', label='leaving' )
plt.legend()
plt.xlim(-2,116)
plt.ylim(0, 45)
plt.ylabel('Age')
plt.title('Player entering and leaving the Top100 list');




### DEAD END ###
# Failed to determine retired players
tab=elo.groupby('name').games.rolling(center=False,window=3).mean()
tab=tab.reset_index()
tab2=tab.groupby(['name']).games.last()
tab2.sort_values();




player=elo.groupby(['name'])          .agg({'birth_year'  : min,
                'rank'        : min,
                'timedelta'   : sum,
                'games'       : sum,
                'ranking_date': min,
                'rating'      : max})\
          .reset_index()
LastRating=elo.groupby(['name'])              .agg({'ranking_date': max})              .reset_index()
player=pd.merge(how='inner', left=player, left_on='name', right=LastRating, right_on='name')
Country=elo.groupby(['name'])           .country.last()           .to_frame().reset_index()
player=pd.merge(left=player,  left_on='name', right=Country,right_on='name')
player.rename(columns = {'games':'TotalGames',
                         'timedelta': 'TimeListed',
                         'birth_year': 'BirthYear',
                         'rank':'BestRank',
                         'ranking_date_x':'FirstListed',
                         'ranking_date_y':'LastListed',
                         'country': 'Country',
                         'rating': 'Rating'},
              inplace=True)
#player['FirstListedNUM'] = round((player.FirstListed.subtract(pd.Timestamp(STARTDATE_)))/pd.Timedelta(1, unit='M'))
#player['LastListedNUM'] = round((player.LastListed.subtract(pd.Timestamp(STARTDATE_)))/pd.Timedelta(1, unit='M'))

# Regions
regions = {
    'Western Europe' :
        ['ENG','SWE','ESP','UKR','FRA','BEL','NOR','GER','SUI','GRE',
         'FIN','AUT','DEN','ISL','NED','IRL'],
    'Eastern Europe':
        ['HUN','POL','SLO','BUL','SVK','CRO','ROM','CZE','MKD','ROU','SRB','BIH'],
    'Former Soviet Union' :
        ['RUS','ARM','BLR','TJK','GEO','MDA','UZB','LAT','KAZ','AZE','LTU'],
    'America' : 
        ['USA','PAR','CAN','CUB','PER','ARG','BRA','VEN'],
    'Middle East & Africa' :
        ['EGY','TUR','ISR','UAE','MAR','IRI'],
    'Asia' :
        ['CHN','PHI','VIE','SGP','SCG','IND','INA']}
regionsmap = {}
for region, countrylist in regions.items():
    regionsmap.update({country: region for country in countrylist})
player['Region']=player['Country'].map(regionsmap)

#print(len(player)) #=334 as crosscheck
player.head()




player.TimeListed.max()




plt.scatter(y=player['TotalGames'], x=player['TimeListed'], color='lightblue');
plt.axvline(x=206, ymin=0, ymax=player['TotalGames'].max(), linewidth=1, color='red', alpha=0.5);
plt.xlim(-5, 210);
plt.title('Games played over time by player');




plt.scatter( y=player['TotalGames'], x=player['BirthYear'], color='lightblue');
plt.xlim(player['BirthYear'].min()-5, player['BirthYear'].max()+5);
plt.ylim(-5, player['TotalGames'].max()+5);
plt.title('Games played by birth year by player');




g=player.groupby(['Country', 'Region'], as_index=False)    ['name'].count()    .rename(columns={'name':'count'})    .sort_values('count', ascending=False)
g['Country2']=''
g.loc[g['count']>4,'Country2']=g['Country']
gg=g.groupby('Region', as_index=False)     ['count'].sum()
    
fig, ax = plt.subplots(1,2,figsize=(14,6))
ax[1].pie(x=gg['count'], labels=gg['Region']);
ax[0].pie(x=g['count'], labels=g['Country2']);
ax[0].set_title('Player by country')
ax[1].set_title('Player by region');




fig, ax = plt.subplots(1,1, figsize=(9,5))
ax = sns.violinplot(x="Region", y="Rating", data=player)
plt.title('Players by region and Max Rating');




sns.heatmap(player.corr(), annot=True)
plt.title('Correlation of player attributes');




#ToDo: How can I fix the random_state ? 
tsne = TSNE(n_components=2, init='pca', random_state=1)
X_tsne = tsne.fit_transform(player[['BestRank', 'TotalGames', 'BirthYear',
                                    'TimeListed', 'Rating']])
clr={'Carlsen, Magnus':'r',
     'So, Wesley':'b',
     'Kramnik, Vladimir':'darkgrey',
     'Caruana, Fabiano':'g',
     'Aronian, Levon':'y',
     'Polgar, Judit': 'pink',
     'Kasparov, Garry': 'k',
     'Tiviakov, Sergei': 'cyan'}

fig, ax  = plt.subplots(1,1,figsize=(17, 6))
ax.scatter(X_tsne[:, 0], X_tsne[:, 1])
for name in clr:
    i=player[player.name==name].index[0]
    plt.scatter(X_tsne[i, 0], X_tsne[i, 1], color=clr[name]) 
    plt.text(s=name, x=X_tsne[i, 0]+10, y=X_tsne[i, 1]+10)
plt.title('t-SNE, hightlight prominent players');   




df=pd.DataFrame(X_tsne, columns=[['X','Y']])
t=df[df['X']==df.X.max()].index
player.loc[t]




def drawRegression(name, ax):
    
    elo_Y = elo[elo.name==name]['rating'].values
    elo_X = elo[elo.name==name]['PubRunMonth']          - min(elo[elo.name==name]['PubRunMonth'])
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(elo_X.values.reshape(-1, 1), elo_Y)
    
    ax.scatter(elo_X,elo_Y,  color='lightblue')
    x=np.linspace(0,max(elo_X),max(elo_X));
    f=regr.coef_*x+regr.intercept_
    ax.plot(f,color='darkblue', alpha=0.5)
    
    ax.set_ylim(2600, 2900)
    ax.set_xlim(-5,210)
    ax.set_title(name)




names=['Carlsen, Magnus'   , 'So, Wesley'       ,'Vachier-Lagrave, Maxime','Nakamura, Hikaru',
       'Anand, Viswanathan', 'Kramnik, Vladimir','Adams, Michael'         ,'Gelfand, Boris',
       'Shirov, Alexei'    , 'Polgar, Judit'    ,'Karpov, Anatoly'        ,'Short, Nigel D.']
fig, ax = plt.subplots(3,4,figsize=(17,10));
for i in range(3):
    for j in range(4):
        drawRegression(names.pop(0),ax=ax[i,j])











