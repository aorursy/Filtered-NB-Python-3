#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

input_dir = '../input/wta/wta/'

# headers coming from https://github.com/JeffSackmann/tennis_wta
HEADERS = ['player_id', 'first_name', 'last_name', 'hand', 'birth_date', 'country_code']

df_players = pd.read_csv(input_dir + "players.csv", low_memory=False, encoding="cp437", names=HEADERS)




df_players.head()




# first line seems to be a dummy one
df_players = df_players.drop(df_players.index[0]).reindex()


# converts date to a understandable format
df_players_hb = df_players[~df_players['birth_date'].isnull()].reset_index()
df_players_hb['birth_date'] = df_players_hb['birth_date'].apply(lambda b: datetime.strptime(str(b), '%Y%m%d.0'))
df_players_hb['birth_year'] = df_players_hb['birth_date'].apply(lambda b: b.year)

df_players_hb.head()




df_players_hb.groupby('birth_year')             .size()             .plot(title="Players birth over the years", 
                  figsize=(20,10)) 




df_players_hb.groupby(['birth_year', 'hand'])             .size()             .unstack('hand')             .replace(np.NaN, 0)             .plot(title="Players' hand over the years", 
                  figsize=(20,10)) 




top_countries = df_players_hb.groupby('country_code').size().sort_values(ascending=False)[:5].index
df_players_hb[df_players_hb['country_code'].isin(top_countries)]             .groupby(['birth_year', 'country_code'])             .size()             .unstack('country_code')             .replace(np.NaN, 0)             .plot(title="Players birth by country over the year", 
                  figsize=(20,10))




df_players.groupby('country_code')             .size()             .sort_values(ascending=False)[:10]             .plot.bar(title="Top 10 countries with the most player", 
                  figsize=(20,10))




df_players.groupby('first_name')             .size()             .sort_values(ascending=False)[:10]             .plot.bar(title="Common player names among tennis players", 
                  figsize=(20,10))

