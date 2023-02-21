#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
from numpy import random




#Data loading
con = sqlite3.connect('../input/database.sqlite')
countries = pd.read_sql_query("SELECT * from Country", con)
matches = pd.read_sql_query("SELECT * from Match", con)
leagues = pd.read_sql_query("SELECT * from League", con)
teams = pd.read_sql_query("SELECT * from Team", con)




type(con)




avg_home_goals = matches.home_team_goal.mean(axis=0)
avg_away_goals = matches.away_team_goal.mean(axis=0)




print(avg_home_goals,avg_away_goals)




y1 = matches.home_team_goal
y2 = matches.away_team_goal




plt.hist(y1)
plt.hist(y2, alpha = 0.5)




matches.iloc[]




countries.head()




eng_matches = matches[matches.country_id == 1729]




eng_matches




eng_matches.iloc[0]




y1 = matches.home_team_goal - matches.away_team_goal
y2 = 1/matches.B365H
x = matches.index




plt.plot(x,y1/10,'ro')
plt.plot(x,y2, 'bo')




np.corrcoef(y1,y2)




y1.head()






