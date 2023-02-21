#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.




import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
table_names = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())




player_table = pd.read_sql_query("SELECT * FROM Player", con)
player_att_table = pd.read_sql_query("SELECT * FROM Player_Attributes", con)
match_table = pd.read_sql_query("SELECT * FROM Match", con)
league_table = pd.read_sql_query("SELECT * FROM League", con)
country_table = pd.read_sql_query("SELECT * FROM Country", con)
team_table = pd.read_sql_query("SELECT * FROM Team", con)
team_att_table = pd.read_sql_query("SELECT * FROM Team_Attributes", con)




print("Dimension of Country Table is: {}".format(country_table.shape))
print(100*"*")
print(country_table.info())
print(100*"*")
print(country_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(country_table.describe())
print(100*"*")
print(country_table.isnull().sum(axis=0))




country_table




print("Dimension of League Table is: {}".format(league_table.shape))
print(100*"*")
print(league_table.info())
print(100*"*")
print(league_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(league_table.describe())
print(100*"*")
print(league_table.isnull().sum(axis=0))




league_table




print("Dimension of Player Table is: {}".format(player_table.shape))
print(100*"*")
print(player_table.info())
print(100*"*")
print(player_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(player_table.describe())
print(100*"*")
print(player_table.isnull().sum(axis=0))
#Player table has no missing data




fig1, ax1 = plt.subplots(nrows = 1, ncols = 2)
fig1.set_size_inches(14,4)
sns.boxplot(data = player_table.loc[:,["height",'weight']], ax = ax1[0])
ax1[0].set_xlabel('Player Table Features')
ax1[0].set_ylabel('')
sns.distplot(a = player_table.loc[:,["height"]], bins= 10, kde = True, ax = ax1[1],             label = 'Height')
sns.distplot(a = player_table.loc[:,["weight"]], bins= 10, kde = True, ax = ax1[1],             label = 'Weight')
ax1[1].legend()
sns.jointplot(x='height',y = 'weight',data = player_table,kind = 'scatter')
fig1.tight_layout()




print("Cardinality of Feature: Height - {:0.3f}%".format(         100 * (len(np.unique(player_table.loc[:,'height'])) / len(player_table.loc[:,'height']))))
print("Cardinality of Feature: Weight - {:0.3f}%".format(         100 * (len(np.unique(player_table.loc[:,'weight'])) / len(player_table.loc[:,'weight']))))




print("Dimension of Player Attributes Table is: {}".format(player_att_table.shape))
print(100*"*")
print(player_att_table.info())
print(100*"*")
print(player_att_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(player_att_table.describe())
print(100*"*")
print(player_att_table.isnull().sum(axis=0))
#Player Attributes Table has some missing data




np.unique(player_att_table.dtypes.values)




player_att_table.select_dtypes(include =['float64','int64']).head().loc[:,player_att_table.select_dtypes(include =['float64','int64']).columns[3:]].head()




corr2 = player_att_table.select_dtypes(include =['float64','int64']).loc[:,player_att_table.select_dtypes(include =['float64','int64']).columns[3:]].corr()




fig2,ax2 = plt.subplots(nrows = 1,ncols = 1)
fig2.set_size_inches(w=24,h=24)
sns.heatmap(corr2,annot = True,linewidths=0.5,ax = ax2)




fig3, ax3 = plt.subplots(nrows = 1, ncols = 3)
fig3.set_size_inches(12,4)
sns.countplot(x = player_att_table['preferred_foot'],ax = ax3[0])
sns.countplot(x = player_att_table['attacking_work_rate'],ax = ax3[1])
sns.countplot(x = player_att_table['defensive_work_rate'],ax = ax3[2])
fig3.tight_layout()




print(player_att_table['attacking_work_rate'].value_counts())
print(100*'*')
print(player_att_table['defensive_work_rate'].value_counts())
print(100*'*')
print(player_att_table.shape)




player_att_table.loc[~(player_att_table['attacking_work_rate'].                                                  isin(['medium','high','low'])                       | player_att_table['defensive_work_rate'].isin(['medium','high','low'])),:].head()




player_att_table_updated1 = player_att_table.loc[(player_att_table['attacking_work_rate'].                                                  isin(['medium','high','low'])                       & player_att_table['defensive_work_rate'].isin(['medium','high','low'])),:]
print(player_att_table_updated1.shape)
player_att_table_updated1.head()




fig4, ax4 = plt.subplots(nrows = 1, ncols = 3)
fig4.set_size_inches(12,3)
sns.countplot(x = player_att_table_updated1['preferred_foot'],ax = ax4[0])
sns.countplot(x = player_att_table_updated1['attacking_work_rate'],ax = ax4[1])
sns.countplot(x = player_att_table_updated1['defensive_work_rate'],ax = ax4[2])
fig4.tight_layout()




fig4, ax4 = plt.subplots(nrows = 1, ncols = 3)
fig4.set_size_inches(12,3)
sns.barplot(x ='preferred_foot', y = 'preferred_foot', data = player_att_table_updated1,            estimator = lambda x: len(x)/len(player_att_table_updated1) * 100, ax = ax4[0],           orient = 'v')
ax4[0].set(ylabel = 'Percentage',title = 'Preferred Foot')
sns.barplot(x ='attacking_work_rate', y = 'attacking_work_rate', data = player_att_table_updated1,            estimator = lambda x: len(x)/len(player_att_table_updated1) * 100, ax = ax4[1],           orient = 'v')
ax4[1].set(ylabel = 'Percentage',title = 'Attacking Work Rate')
sns.barplot(x ='defensive_work_rate', y = 'defensive_work_rate', data = player_att_table_updated1,            estimator = lambda x: len(x)/len(player_att_table_updated1) * 100, ax = ax4[2],           orient = 'v')
ax4[2].set(ylabel = 'Percentage',title = 'Defensive Work Rate')
fig4.tight_layout()




att_work_rate = player_att_table_updated1.groupby('attacking_work_rate').size().values.tolist()
def_work_rate = player_att_table_updated1.groupby('defensive_work_rate').size().values.tolist()




print("Attacking work rate factor, Medium, accounts for: {:0.3f}% of features".format(100 * att_work_rate[2]/np.sum(att_work_rate)))
print("Defensive work rate factor, Medium, accounts for: {:0.3f}% of features".format(100 * def_work_rate[2]/np.sum(def_work_rate)))




print("Percentage of instances removed from player attributes table: {:0.2f}%".      format(100* (1 - player_att_table_updated1.shape[0]/player_att_table.shape[0])))
print("We removed {} instances from Player Attributes table".      format(-player_att_table_updated1.shape[0] + player_att_table.shape[0]))




print("Dimension of Player Attributes Table Updated 1 is: {}".format(player_att_table_updated1.shape))
print(100*"*")
print(player_att_table_updated1.info())
print(100*"*")
print(player_att_table_updated1.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(player_att_table_updated1.describe())
print(100*"*")
print(player_att_table_updated1.isnull().sum(axis=0))
#No more missing data




pat = player_att_table_updated1.loc[:,player_att_table_updated1.columns.tolist()[3:]]




fig5, ax5 = plt.subplots(nrows=5,ncols=7)
fig5.set_size_inches(16,12)
for i,j in enumerate(player_att_table_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.distplot(pat.loc[:,j],kde = False,hist = True, ax = ax5[int(i/7)][i%7])
fig5.tight_layout()




fig6, ax6 = plt.subplots(nrows=5,ncols=7)
fig6.set_size_inches(16,12)
for i,j in enumerate(player_att_table_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "preferred_foot", y = j, data= pat, ax = ax6[int(i/7)][i%7])
fig6.tight_layout()




fig7, ax7 = plt.subplots(nrows=5,ncols=7)
fig7.set_size_inches(16,12)
for i,j in enumerate(player_att_table_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "attacking_work_rate", y = j, data= pat, ax = ax7[int(i/7)][i%7])
fig7.tight_layout() 




fig8, ax8 = plt.subplots(nrows=5,ncols=7)
fig8.set_size_inches(16,12)
for i,j in enumerate(player_att_table_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "defensive_work_rate", y = j, data= pat, ax = ax8[int(i/7)][i%7])
fig8.tight_layout()




print("Dimension of Team Table is: {}".format(team_table.shape))
print(100*"*")
print(team_table.info())
print(100*"*")
print(team_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(team_table.describe())
print(100*"*")
print(team_table.isnull().sum(axis=0))




team_table[team_table.loc[:,'team_fifa_api_id'].isnull()]




team_table_updated = team_table[~team_table.loc[:,'team_fifa_api_id'].isnull()]




print("Dimension of Team Table Updated is: {}".format(team_table_updated.shape))
print(100*"*")
print(team_table_updated.info())
print(100*"*")
print(team_table_updated.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(team_table_updated.describe())
print(100*"*")
print(team_table_updated.isnull().sum(axis=0))
print(100*"*")
print(team_table_updated.select_dtypes(exclude=['float64','int64']).apply(lambda x: len(x.unique().tolist()),axis = 0))




print(len(team_table_updated['team_long_name'].unique().tolist()),      len(team_table_updated['team_short_name'].unique().tolist()))




my_team = dict()
for i,j in list(team_table_updated.iloc[:,3:].groupby('team_short_name')):
    my_team[i] = j.iloc[:,0].values.tolist()




{k:v for k,v in my_team.items() if len(v) > 1}
#List of teams with similar short team names




print("Dimension of Team Attributes Table is: {}".format(team_att_table.shape))
print(100*"*")
print(team_att_table.info())
print(100*"*")
print(team_att_table.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(team_att_table.describe())
print(100*"*")
print(team_att_table.isnull().sum(axis=0))




team_att_table.loc[team_att_table['buildUpPlayDribbling'].isnull(),:].head()




team_att_table.loc[~team_att_table['buildUpPlayDribbling'].isnull(),:].head()




team_att_table_updated1 = team_att_table.drop(['buildUpPlayDribbling'],axis = 1)
print("Dimension of Team Attributes Table updated is: {}".format(team_att_table_updated1.shape))
print(100*"*")
print(team_att_table_updated1.info())
print(100*"*")
print(team_att_table_updated1.select_dtypes(exclude=['float64','int64']).describe())
print(100*"*")
print(team_att_table_updated1.describe())
print(100*"*")
print(team_att_table_updated1.isnull().sum(axis=0))




tat = team_att_table_updated1.loc[:,team_att_table_updated1.columns.tolist()[3:]]




sns.pairplot(tat)
#Little to no correlation beween any of the continuous features




fig9, ax9 = plt.subplots(nrows=2,ncols=4)
fig9.set_size_inches(12,6)
for i,j in enumerate(team_att_table_updated1.select_dtypes(include = ['int64']).columns[3:].tolist()):
    sns.distplot(tat.loc[:,j],kde =True,hist = True, ax = ax9[int(i/4)][i%4])
fig9.tight_layout()




team_att_table_updated1.select_dtypes(include = ['int64']).head()




sns.boxplot(data = team_att_table_updated1.select_dtypes(include = ['int64']).iloc[:,3:],           orient = 'h')




fig9, ax9 = plt.subplots(nrows=3,ncols=4)
fig9.set_size_inches(14,8)
for i,j in enumerate(team_att_table_updated1.select_dtypes(include = ['object']).columns[1:].tolist()):
    #sns.countplot(tat.loc[:,j], ax = ax9[int(i/4)][i%4])
    sns.barplot(x = j, y = j, data = tat,            estimator = lambda x: len(x)/len(tat) * 100, ax = ax9[int(i/4)][i%4],           orient = 'v')
    ax9[int(i/4)][i%4].set(xlabel = "")
fig9.tight_layout()




tat.select_dtypes(include = ['int64']).columns.tolist()




sns.pairplot(tat,hue = tat.select_dtypes(include = ['object']).          columns.tolist()[1]) 




sns.pairplot(tat,hue = tat.select_dtypes(include = ['object']).          columns.tolist()[12]) 




fig9, ax9 = plt.subplots(nrows=2,ncols=4)
fig9.set_size_inches(12,6)
for i,j in enumerate(team_att_table_updated1.select_dtypes(include = ['int64']).columns[3:].tolist()):
    sns.boxplot(data = tat, y = j, x = tat.select_dtypes(include = ['object']).columns[3],                                                      ax = ax9[int(i/4)][i%4])
fig9.tight_layout()

