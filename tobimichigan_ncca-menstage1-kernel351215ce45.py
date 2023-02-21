#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import csv
import math
import os
import pandas as pd
import random
import sklearn
from sklearn.model_selection import KFold
from sklearn import linear_model
#from sklearn import cross_validation, linear_model
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import datetime, time
import sys
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_fdr ="/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/"




if len(sys.argv) == 1:
    print("- - - - - - - - - - - - - - - - - - - - - - - - -")
    print("NO DEFINED YEAR! QUITTING!")
    print(str(time.time()))
    print("Use 'python3 NCAA.py [YEAR]' to define the year.")
    print("- - - - - - - - - - - - - - - - - - - - - - - - -")
    quit()
else:
    theYear = sys.argv[1]

print("Generating results for " + theYear +".")




seasondata=pd.read_csv(df_fdr +'MSeasons.csv')
seasondata.tail()




# Total held seasons including the current
seasondata['Season'].count()




mteams=pd.read_csv(df_fdr +'MTeams.csv')
mteams.head()




mseeds=pd.read_csv(df_fdr +'MNCAATourneySeeds.csv')
mseeds.head()




mseeds = pd.merge(mseeds, mteams,on='TeamID')
mseeds.head()




# Spliting regions from the Seeds

mseeds['Region'] = mseeds['Seed'].apply(lambda x: x[0][:1])
mseeds['Seed'] = mseeds['Seed'].apply(lambda x: int(x[1:3]))
print(mseeds.head())
print(mseeds.shape)




# Teams with maximum top seeds
fig = plt.gcf()
fig.set_size_inches(10, 6)
colors = ['dodgerblue', 'plum', '#F0A30A','#8c564b','orange','green','yellow'] 

mseeds[mseeds['Seed'] ==1]['TeamName'].value_counts()[:10].plot(kind='bar',color=colors,linewidth=2,edgecolor='black')
plt.xlabel('Number of times in Top seeded positions')




mcompactre = pd.read_csv(df_fdr +'MRegularSeasonCompactResults.csv')
mcompactre.head()




#  Score Avergae for Win vs Loss over a certain past
x = mcompactre.groupby('Season')[['WScore','LScore']].mean()

fig = plt.gcf()
fig.set_size_inches(28, 12)
plt.plot(x.index,x['WScore'],marker='o', markerfacecolor='green', markersize=12, color='green', linewidth=4)
plt.plot(x.index,x['LScore'],marker=7, markerfacecolor='red', markersize=12, color='red', linewidth=4)
plt.legend()




#df_fdr = '/kaggle/working/'




files = os.listdir(df_fdr)
names = [x.split('.')[0] for x in files]
path_list = [f'{df_fdr}{x}' for x in files]
df_dict = {}
for num, name in enumerate(names):
    try:
        df_dict[name] = pd.read_csv(path_list[num])
    except:
        print(f'{name} did not load')




print('Summary of files:\n___________________________')
for name, df in df_dict.items():
    print(f'{name}: {df.shape}')




## > DATA
path_datasets = df_fdr

df_rs_c_res = pd.read_csv(path_datasets + 'MRegularSeasonCompactResults.csv')
df_rs_d_res = pd.read_csv(path_datasets + 'MRegularSeasonDetailedResults.csv')
df_teams = pd.read_csv(path_datasets + 'MTeams.csv')
df_seeds = pd.read_csv(path_datasets + 'MNCAATourneySeeds.csv')
coaches = pd.read_csv(path_datasets + 'MTeams.csv')
df_tourn = pd.read_csv(path_datasets + 'MNCAATourneyCompactResults.csv')




## > DATA CLEANING
# clean team information

df_teams_cl = df_teams.iloc[:,:2]

## > DATA CLEANING
# clean seed information

df_seeds_cl = df_seeds.loc[:, ['TeamID', 'Season', 'Seed']]

def clean_seed(seed):
    s_int = int(seed[1:3])
    return s_int

def extract_seed_region(seed):
    s_reg = seed[0:1]
    return s_reg

df_seeds_cl['seed_int'] = df_seeds_cl['Seed'].apply(lambda x: clean_seed(x))
df_seeds_cl['seed_region'] = df_seeds_cl['Seed'].apply(lambda x: extract_seed_region(x))
df_seeds_cl['top_seeded_teams'] = np.where(df_seeds_cl['Seed'].isnull(), 0, 1)

df_seeds_cl.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds_cl.head()




## > DATA CLEANING
# create games dataframe WINNERS

def new_name_w_1(old_name):
    match = re.match(r'^L', old_name)
    if match:
        out = re.sub('^L','', old_name)
        return out + '_opp'
    return old_name

def new_name_w_2(old_name):
    match = re.match(r'^W', old_name)
    if match:
        out = re.sub('^W','', old_name)
        return out
    return old_name

def prepare_stats_extended_winners(df_in, df_seed_in, df_teams_in):
    df_in['poss'] = df_in['WFGA'] + 0.475*df_in['WFTA'] - df_in['WOR'] + df_in['WTO']
    df_in['opp_poss'] = df_in['LFGA'] + 0.475*df_in['LFTA'] - df_in['LOR'] + df_in['LTO']
    df_in['off_rating'] = 100*(df_in['WScore'] / df_in['poss'])
    df_in['def_rating'] = 100*(df_in['LScore'] / df_in['opp_poss'])
    df_in['net_rating'] = df_in['off_rating'] - df_in['def_rating']
    df_in['pace'] = 48*((df_in['poss']+df_in['opp_poss'])/(2*(240/5)))
    
    df_in = df_in.rename(columns={'WTeamID':'TeamID', 
                                  'WLoc':'_Loc',
                                  'LTeamID':'TeamID_opp',
                                  'WScore':'Score_left', 
                                  'LScore':'Score_right'})
    
    df_seeds_opp = df_seed_in.rename(columns={'TeamID':'TeamID_opp',
                                              'seed_int':'seed_int_opp',
                                              'seed_region':'seed_region_opp',
                                              'top_seeded_teams':'top_seeded_teams_opp'})
    
    df_out = pd.merge(left=df_in, right=df_seeds_cl, how='left', on=['Season', 'TeamID'])
    df_out = pd.merge(left=df_out, right=df_seeds_opp, how='left', on=['Season', 'TeamID_opp'])
    df_out = pd.merge(left=df_out, right=df_teams_in, how='left', on=['TeamID'])
    
    df_out['DayNum'] = pd.to_numeric(df_out['DayNum'])
    df_out['win_dummy'] = 1
    
    df_out['seed_int'] = np.where(df_out['seed_int'].isnull(), 20, df_out['seed_int'])
    df_out['seed_region'] = np.where(df_out['seed_region'].isnull(), 'NoTour', df_out['seed_region'])
    df_out['top_seeded_teams'] = np.where(df_out['top_seeded_teams'].isnull(), 0, df_out['top_seeded_teams'])
    
    df_out['seed_int_opp'] = np.where(df_out['seed_int_opp'].isnull(), 20, df_out['seed_int_opp'])
    df_out['seed_region_opp'] = np.where(df_out['seed_region_opp'].isnull(), 'NoTour', df_out['seed_region_opp'])
    df_out['top_seeded_teams_opp'] = np.where(df_out['top_seeded_teams_opp'].isnull(), 0, df_out['top_seeded_teams_opp'])
    
    df_out = df_out.rename(columns=new_name_w_1)
    df_out = df_out.rename(columns=new_name_w_2)
    
    return df_out

df_games_w = prepare_stats_extended_winners(df_rs_d_res, df_seeds_cl, df_teams_cl)

df_games_w.head()




## > DATA CLEANING
# create games dataframe LOSERS

def new_name_l_1(old_name):
    match = re.match(r'^W', old_name)
    if match:
        out = re.sub('^W','', old_name)
        return out + '_opp'
    return old_name

def new_name_l_2(old_name):
    match = re.match(r'^L', old_name)
    if match:
        out = re.sub('^L','', old_name)
        return out
    return old_name

def prepare_stats_extended_losers(df_in, df_seed_in, df_teams_in):
    df_in['poss'] = df_in['LFGA'] + (0.475*df_in['LFTA']) - df_in['LOR'] + df_in['LTO']
    df_in['opp_poss'] = df_in['WFGA'] + (0.475*df_in['WFTA']) - df_in['WOR'] + df_in['WTO']
    df_in['off_rating'] = 100*(df_in['LScore'] / df_in['poss'])
    df_in['def_rating'] = 100*(df_in['WScore'] / df_in['opp_poss'])
    df_in['net_rating'] = df_in['off_rating'] - df_in['def_rating']
    df_in['pace'] = 48*((df_in['poss']+df_in['opp_poss'])/(2*(240/5)))
    
    df_in = df_in.rename(columns={'LTeamID':'TeamID', 
                                  'LLoc':'_Loc',
                                  'WTeamID':'TeamID_opp',
                                  'LScore':'Score_left', 
                                  'WScore':'Score_right'})
    
    df_seeds_opp = df_seed_in.rename(columns={'TeamID':'TeamID_opp',
                                              'seed_int':'seed_int_opp',
                                              'seed_region':'seed_region_opp',
                                              'top_seeded_teams':'top_seeded_teams_opp'})
    
    df_out = pd.merge(left=df_in, right=df_seeds_cl, how='left', on=['Season', 'TeamID'])
    df_out = pd.merge(left=df_out, right=df_seeds_opp, how='left', on=['Season', 'TeamID_opp'])
    df_out = pd.merge(left=df_out, right=df_teams_in, how='left', on=['TeamID'])
    
    df_out['DayNum'] = pd.to_numeric(df_out['DayNum'])
    df_out['win_dummy'] = 0
    
    df_out['seed_int'] = np.where(df_out['seed_int'].isnull(), 20, df_out['seed_int'])
    df_out['seed_region'] = np.where(df_out['seed_region'].isnull(), 'NoTour', df_out['seed_region'])
    df_out['top_seeded_teams'] = np.where(df_out['top_seeded_teams'].isnull(), 0, df_out['top_seeded_teams'])
    
    df_out['seed_int_opp'] = np.where(df_out['seed_int_opp'].isnull(), 20, df_out['seed_int_opp'])
    df_out['seed_region_opp'] = np.where(df_out['seed_region_opp'].isnull(), 'NoTour', df_out['seed_region_opp'])
    df_out['top_seeded_teams_opp'] = np.where(df_out['top_seeded_teams_opp'].isnull(), 0, df_out['top_seeded_teams_opp'])

    df_out = df_out.rename(columns=new_name_l_1)
    df_out = df_out.rename(columns=new_name_l_2)
    
    return df_out

df_games_l = prepare_stats_extended_losers(df_rs_d_res, df_seeds_cl, df_teams_cl)

df_games_l.head()




## > MERGE

df_games_t = pd.concat([df_games_w,df_games_l], sort=True)

## > AGGREGATED STATS BY TEAM AND SEASON

def aggr_stats(df):
    d = {}
    d['G'] = df['win_dummy'].count()
    d['W'] = df['win_dummy'].sum()
    d['L'] = np.sum(df['win_dummy'] == 0)
    d['G_vs_topseeds'] = np.sum(df['top_seeded_teams_opp'] == 1)
    d['W_vs_topseeds'] = np.sum((df['win_dummy'] == 1) & (df['top_seeded_teams_opp'] == 1))
    d['L_vs_topseeds'] = np.sum((df['win_dummy'] == 0) & (df['top_seeded_teams_opp'] == 1))
    d['G_last30D'] = np.sum((df['DayNum'] > 100))
    d['W_last30D'] = np.sum((df['win_dummy'] == 1) & (df['DayNum'] > 100))
    d['L_last30D'] = np.sum((df['win_dummy'] == 0) & (df['DayNum'] > 100))
    d['G_H'] = np.sum((df['_Loc'] == 'H'))
    d['W_H'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'H'))
    d['L_H'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'H'))
    d['G_A'] = np.sum((df['_Loc'] == 'A'))
    d['W_A'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'A'))
    d['L_A'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'A'))
    d['G_N'] = np.sum((df['_Loc'] == 'N'))
    d['W_N'] = np.sum((df['win_dummy'] == 1) & (df['_Loc'] == 'N'))
    d['L_N'] = np.sum((df['win_dummy'] == 0) & (df['_Loc'] == 'N'))
    
    d['PS'] = np.mean(df['Score_left'])
    d['PS_H'] = np.mean(df['Score_left'][df['_Loc'] == 'H'])
    d['PS_A'] = np.mean(df['Score_left'][df['_Loc'] == 'A'])
    d['PS_N'] = np.mean(df['Score_left'][df['_Loc'] == 'N'])
    d['PS_last30D'] = np.mean(df['Score_left'][df['DayNum'] > 100])
    
    d['PA'] = np.mean(df['Score_right'])
    d['PA_H'] = np.mean(df['Score_right'][df['_Loc'] == 'H'])
    d['PA_A'] = np.mean(df['Score_right'][df['_Loc'] == 'A'])
    d['PA_N'] = np.mean(df['Score_right'][df['_Loc'] == 'N'])
    d['PA_last30D'] = np.mean(df['Score_right'][df['DayNum'] > 100])
    
    d['poss_m'] = np.mean(df['poss'])
    d['opp_poss_m'] = np.mean(df['opp_poss'])
    d['off_rating_m'] = np.mean(df['off_rating'])
    d['def_rating_m'] = np.mean(df['def_rating'])
    d['net_rating_m'] = np.mean(df['net_rating'])
    d['pace_m'] = np.mean(df['pace'])
    
    d['off_rating_m_last30D'] = np.mean(df['off_rating'][df['DayNum'] > 100])
    d['def_rating_m_last30D'] = np.mean(df['def_rating'][df['DayNum'] > 100])
    d['net_rating_m_last30D'] = np.mean(df['net_rating'][df['DayNum'] > 100])
    
    d['off_rating_m_vs_topseeds'] = np.mean(df['off_rating'][df['top_seeded_teams_opp'] == 1])
    d['def_rating_m_vs_topseeds'] = np.mean(df['def_rating'][df['top_seeded_teams_opp'] == 1])
    d['net_rating_m_vs_topseeds'] = np.mean(df['net_rating'][df['top_seeded_teams_opp'] == 1])
    
    return pd.Series(d)


df_agg_stats = df_games_t.                          groupby([df_games_t['Season'], 
                                   df_games_t['TeamID'],
                                   df_games_t['TeamName'],
                                   df_games_t['seed_int'],
                                   df_games_t['seed_region']], 
                                  as_index=False).\
                          apply(aggr_stats).\
                          reset_index()


df_agg_stats['w_pct'] = df_agg_stats['W'] / df_agg_stats['G']
df_agg_stats['w_pct_last30D'] = df_agg_stats['W_last30D'] / df_agg_stats['G_last30D']
df_agg_stats['w_pct_vs_topseeds'] = df_agg_stats['W_vs_topseeds'] / df_agg_stats['G_vs_topseeds']

df_agg_stats.head(20)




## > DATA CLEANING 

# prepare tournament dataset
def prepare_tournament_datasets(df_tourn_in, df_agg_stats_in):
    
    df_tourn_in['TeamID'] = df_tourn_in[['WTeamID','LTeamID']].min(axis=1)
    df_tourn_in['TeamID_opp'] = df_tourn_in[['WTeamID','LTeamID']].max(axis=1)
    df_tourn_in['win_dummy'] = np.where(df_tourn_in['TeamID'] == df_tourn_in['WTeamID'], 1, 0)
    df_tourn_in['delta'] = np.where(df_tourn_in['win_dummy'] == 1,
                                    df_tourn_in['WScore'] - df_tourn['LScore'],
                                    df_tourn_in['LScore'] - df_tourn['WScore'])
    df_tourn_in['Score_left'] = np.where(df_tourn_in['win_dummy'] == 1,
                                         df_tourn_in['WScore'],
                                         df_tourn_in['LScore'])
    df_tourn_in['Score_right'] = np.where(df_tourn_in['win_dummy'] == 1,
                                          df_tourn_in['LScore'],
                                          df_tourn_in['WScore'])
                                 
    df_teams_gr_left = df_agg_stats_in.loc[:,['Season', 'TeamID',
                                              'w_pct', 'seed_int', 
                                              'net_rating_m_last30D',
                                              'net_rating_m_vs_topseeds',
                                              'net_rating_m']].\
                  rename(columns={'w_pct':'w_pct_left',
                                  'seed_int':'seed_int_left', 
                                  'net_rating_m_last30D':'net_rating_m_last30D_left', 
                                  'net_rating_m_vs_topseeds':'net_rating_m_vs_topseeds_left', 
                                  'net_rating_m':'net_rating_m_left'})
    
    df_teams_gr_right = df_agg_stats_in.loc[:,['Season', 'TeamID',
                                               'w_pct', 'seed_int',
                                               'net_rating_m_last30D',
                                               'net_rating_m_vs_topseeds',
                                               'net_rating_m']].\
                  rename(columns={'TeamID':'TeamID_opp',
                                  'w_pct':'w_pct_right',
                                  'seed_int':'seed_int_right', 
                                  'net_rating_m_last30D':'net_rating_m_last30D_right', 
                                  'net_rating_m_vs_topseeds':'net_rating_m_vs_topseeds_right', 
                                  'net_rating_m':'net_rating_m_right'})
    
    df_tourn_out = pd.merge(left=df_tourn_in, 
                            right=df_teams_gr_left, 
                            how='left', on=['Season', 'TeamID'])
    df_tourn_out = pd.merge(left=df_tourn_out, 
                            right=df_teams_gr_right, 
                            how='left', on=['Season', 'TeamID_opp'])

    df_tourn_out['delta_w_pct'] = df_tourn_out['w_pct_left'] -                                          df_tourn_out['w_pct_right']


    df_tourn_out['delta_seed_int'] = df_tourn_out['seed_int_left'] -                                           df_tourn_out['seed_int_right']


    df_tourn_out['delta_net_rating_m'] = df_tourn_out['net_rating_m_left'] - df_tourn_out['net_rating_m_right']
    
    df_tourn_out['delta_net_rating_m_last30D'] = df_tourn_out['net_rating_m_last30D_left'] - df_tourn_out['net_rating_m_last30D_right']
    
    df_tourn_out['delta_net_rating_m_vs_topseeds'] = df_tourn_out['net_rating_m_vs_topseeds_left'] - df_tourn_out['net_rating_m_vs_topseeds_right']
    
    df_out = df_tourn_out.loc[:, ['Season', 'DayNum',
                                  'TeamID', 'TeamID_opp',
                                  'Score_left', 'Score_right',
                                  'win_dummy', 
                                  'delta', 'NumOT', 'delta_w_pct', 
                                  'delta_net_rating_m_last30D',
                                  'delta_net_rating_m_vs_topseeds',
                                  'delta_net_rating_m', 'delta_seed_int']]
                                    
    return df_out

                                    
df_tourn_cl = prepare_tournament_datasets(df_tourn, df_agg_stats)                                    
df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].head(10)




## > DUKE RS
df_agg_stats[(df_agg_stats['TeamName'] == 'Duke') & (df_agg_stats['Season'] == 2019)].head()




## > DUKE TOURNAMENT
df_tourn_cl[((df_tourn_cl['TeamID'] == 1181) | (df_tourn_cl['TeamID_opp'] == 1181)) &             (df_tourn_cl['Season'] == 2019)].head(10)




## > DATA VIZ RS
sns.set(style="ticks", color_codes=True)

df_teams_gr = df_agg_stats.loc[:,['w_pct',
                                  'net_rating_m', 'net_rating_m_last30D', 
                                  'net_rating_m_vs_topseeds', 'pace_m']]

df_teams_gr = df_teams_gr.fillna(0)

#df_teams_gr.describe()
sns.pairplot(df_teams_gr, palette="Set1")




## > DATA VIZ TOURNEY
sns.set(style="ticks", color_codes=True)

df_tourn_cl_gr = df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].reindex()

df_tourn_cl_gr = df_tourn_cl_gr.loc[:,['win_dummy',
                                       'delta_net_rating_m_last30D',
                                       'delta_net_rating_m_vs_topseeds',
                                       'delta_net_rating_m',  
                                       'delta_seed_int']]

fig, ax = plt.subplots(figsize=(11, 7))
sns.boxplot(x="variable", y="value", hue = 'win_dummy', ax=ax, 
            data=pd.melt(df_tourn_cl_gr, id_vars='win_dummy'), palette="Set2")
plt.xticks(rotation=45)




## > DATA VIZ TOURNEY
df_tourn_cl_gr = df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].reindex()

df_tourn_cl_gr = df_tourn_cl_gr.loc[:,['win_dummy',
                                       'delta_w_pct']]

fig, ax = plt.subplots(figsize=(9, 7))
sns.boxplot(x="variable", y="value", hue = 'win_dummy', ax=ax, 
            data=pd.melt(df_tourn_cl_gr, id_vars='win_dummy'), palette="Set2")




## > Correlation
# Compute the correlation matrix
df_tourn_cl_gr = df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018]))].reindex()

df_tourn_cl_gr = df_tourn_cl_gr.loc[:,['win_dummy',
                                       'delta_net_rating_m_last30D',
                                       'delta_net_rating_m_vs_topseeds',                                       
                                       'delta_net_rating_m',  
                                       'delta_w_pct',
                                       'delta_seed_int']].fillna(0)

corr = df_tourn_cl_gr.corr()
fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(corr, cmap="YlGnBu", ax = ax)




## > AR
def somers2_py(x, y):
    
    from sklearn.metrics import roc_auc_score
    
    C = roc_auc_score(y, x)
    Dxy = (2 * roc_auc_score(y, x))  - 1
    
    return Dxy, C

def apply_somers(df):
    
    d = {}
    
    dxy, cxy = somers2_py(df['value'],
                          df['win_dummy'])
    
    d['Dxy'] = dxy
    d['C'] = cxy
    
    
    return pd.Series(d)

df_tourn_cl_gr = df_tourn_cl[(df_tourn_cl['Season'].isin([2015, 2016, 2017, 2018,2019]))].reindex()

df_tourn_cl_gr = df_tourn_cl_gr.loc[:,['win_dummy',
                                       'delta_net_rating_m_last30D',
                                       'delta_net_rating_m_vs_topseeds',                                       
                                       'delta_net_rating_m',  
                                       'delta_w_pct',
                                       'delta_seed_int']].fillna(0)

df_ar = pd.melt(df_tourn_cl_gr, id_vars='win_dummy')

df_ar.groupby(['variable']).                          apply(apply_somers).                          reset_index().                          sort_values(by=['Dxy'], ascending=False)




timeString = str(time.time())
base_elo = 1600
team_elos = {}  # Reset each year.
team_stats = {}
X = []
y = []
submission_data = []
prediction_year = int(2019)
folder = df_fdr




def calc_elo(win_team, lose_team, season):
    winner_rank = get_elo(season, win_team)
    loser_rank = get_elo(season, lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank


def initialize_data():
    for i in range(1985, int(2019)+1):
        team_elos[i] = {}
        team_stats[i] = {}


def get_elo(season, team):
    try:
        return team_elos[season][team]
    except:
        try:
            # Get the previous season's ending value.
            team_elos[season][team] = team_elos[season-1][team]
            return team_elos[season][team]
        except:
            # Get the starter elo.
            team_elos[season][team] = base_elo
            return team_elos[season][team]


def predict_winner(team_1, team_2, model, season, stat_fields):
    features = []

    # Team 1
    features.append(get_elo(season, team_1))
    for stat in stat_fields:
        features.append(get_stat(season, team_1, stat))

    # Team 2
    features.append(get_elo(season, team_2))
    for stat in stat_fields:
        features.append(get_stat(season, team_2, stat))

    return model.predict_proba([features])


def update_stats(season, team, fields):
    """
    This accepts some stats for a team and udpates the averages.

    First, we check if the team is in the dict yet. If it's not, we add it.
    Then, we try to check if the key has more than 5 values in it.
        If it does, we remove the first one
        Either way, we append the new one.
    If we can't check, then it doesn't exist, so we just add this.

    Later, we'll get the average of these items.
    """
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        # Make sure we have the field.
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []
        #Compare the last 10 games.
        if len(team_stats[season][team][key]) >= 10:
            team_stats[season][team][key].pop()
        team_stats[season][team][key].append(value)


def get_stat(season, team, field):
    try:
        l = team_stats[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0




def build_team_dict():
    team_ids = pd.read_csv(folder + 'MTeams.csv')
    team_id_map = {}
    for index, row in team_ids.iterrows():
        team_id_map[row['TeamID']] = row['TeamName']
    return team_id_map




def build_season_data(all_data):
    # Calculate the elo for every game for every team, each season.
    # Store the elo per season so we can retrieve their end elo
    # later in order to predict the tournaments without having to
    # inject the prediction into this loop.
    print("Building season data.")
    for index, row in all_data.iterrows():
        # Used to skip matchups where we don't have usable stats yet.
        skip = 0

        # Get starter or previous elos.
        team_1_elo = get_elo(row['Season'], row['WTeamID'])
        team_2_elo = get_elo(row['Season'], row['LTeamID'])

        # Add 100 to the home team (# taken from Nate Silver analysis.)
        if row['WLoc'] == 'H':
            team_1_elo += 100
        elif row['WLoc'] == 'A':
            team_2_elo += 100

        # We'll create some arrays to use later.
        team_1_features = [team_1_elo]
        team_2_features = [team_2_elo]

        # print("Building arrays out of the stats.")
        # Build arrays out of the stats we're tracking..
        for field in stat_fields:
            team_1_stat = get_stat(row['Season'], row['WTeamID'], field)
            team_2_stat = get_stat(row['Season'], row['LTeamID'], field)
            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1

        if skip == 0:  # Make sure we have stats.
            # Randomly select left and right and 0 or 1 so we can train
            # for multiple classes.
            if random.random() > 0.5:
                X.append(team_1_features + team_2_features)
                y.append(0)
            else:
                X.append(team_2_features + team_1_features)
                y.append(1)

        # AFTER we add the current stuff to the prediction, update for
        # next time. Order here is key so we don't fit on data from the
        # same game we're trying to predict.
        if row['WFTA'] != 0 and row['LFTA'] != 0:
            stat_1_fields = {
                'score': row['WScore'],
                'fgp': row['WFGM'] / row['WFGA'] * 100,
                'fga': row['WFGA'],
                'fga3': row['WFGA3'],
                '3pp': row['WFGM3'] / row['WFGA3'] * 100,
                'ftp': row['WFTM'] / row['WFTA'] * 100,
                'or': row['WOR'],
                'dr': row['WDR'],
                'ast': row['WAst'],
                'to': row['WTO'],
                'stl': row['WStl'],
                'blk': row['WBlk'],
                'pf': row['WPF']
            }
            stat_2_fields = {
                'score': row['LScore'],
                'fgp': row['LFGM'] / row['LFGA'] * 100,
                'fga': row['LFGA'],
                'fga3': row['LFGA3'],
                '3pp': row['LFGM3'] / row['LFGA3'] * 100,
                'ftp': row['LFTM'] / row['LFTA'] * 100,
                'or': row['LOR'],
                'dr': row['LDR'],
                'ast': row['LAst'],
                'to': row['LTO'],
                'stl': row['LStl'],
                'blk': row['LBlk'],
                'pf': row['LPF']
            }
            update_stats(row['Season'], row['WTeamID'], stat_1_fields)
            update_stats(row['Season'], row['LTeamID'], stat_2_fields)
        
        # Now that we've added them, calc the new elo.
        new_winner_rank, new_loser_rank = calc_elo(
            row['WTeamID'], row['LTeamID'], row['Season'])
        team_elos[row['Season']][row['WTeamID']] = new_winner_rank
        team_elos[row['Season']][row['LTeamID']] = new_loser_rank

        # Adding to make the processing look more dramatic.
        print(new_winner_rank, stat_1_fields, team_1_features)
        print(new_loser_rank, stat_2_fields, team_2_features)
        print(" ")

    return X, y





if __name__ == "__main__":
    stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr',
                   'ast', 'to', 'stl', 'blk', 'pf']

    initialize_data()
    season_data = pd.read_csv(df_fdr + 'MRegularSeasonDetailedResults.csv')
    tourney_data = pd.read_csv(df_fdr + 'MNCAATourneyDetailedResults.csv')
    frames = [season_data, tourney_data]
    all_data = pd.concat(frames)

    # Build the working data.
    X, y = build_season_data(all_data)

    # Fit the model.
    print("Fitting on %d samples." % len(X))

    model = linear_model.LogisticRegression()

    # Check accuracy.kfold_5 = KFold(n_splits = numFolds, shuffle=True)

    print("Checking accuracy with cross-validation:")
    print(sklearn.model_selection.cross_val_score(
        model, X, y, cv=10, scoring='accuracy', n_jobs=-1
    ).mean())      
    
    model.fit(X, y)

    # Now predict tournament matchups.
    print("Getting teams.")
    seeds = pd.read_csv(df_fdr + 'MNCAATourneySeeds.csv')
    # for i in range for year:
    tourney_teams = []
    for index, row in seeds.iterrows():
        if row['Season'] == prediction_year:
            tourney_teams.append(row['TeamID'])

    # Build our prediction of every matchup.
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_1 < team_2:
                prediction = predict_winner(
                    team_1, team_2, model, prediction_year, stat_fields)
                label = str(prediction_year) + '_' + str(team_1) + '_' +                     str(team_2)
                submission_data.append([label, prediction[0][0]])

    # Write the results.
    print("Writing %d results." % len(submission_data))
    if not os.path.isdir("results"):
        os.mkdir("results")
    with open('/kaggle/working/results/submission.'+timeString+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'pred'])
        writer.writerows(submission_data)

    # Now so that we can use this to fill out a bracket, create a readable version.
    print("Outputting readable results.")
    team_id_map = build_team_dict()
    readable = []
    less_readable = []  # A version that's easy to look up.
    for pred in submission_data:
        parts = pred[0].split('_')
        less_readable.append(
            [team_id_map[int(parts[1])], team_id_map[int(parts[2])], pred[1]])
        # Order them properly.
        if pred[1] > 0.5:
            winning = int(parts[1])
            losing = int(parts[2])
            proba = pred[1]
        else:
            winning = int(parts[2])
            losing = int(parts[1])
            proba = 1 - pred[1]
        readable.append(
            [
                '%s beats %s: %f' %
                (team_id_map[winning], team_id_map[losing], proba)
            ]
        )
    with open('/kaggle/working/predictions.'+timeString+'.txt', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(readable)
    with open('/kaggle/working/predictions.'+timeString+'.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(less_readable)

