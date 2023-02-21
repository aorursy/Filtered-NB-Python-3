#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')




df = pd.read_csv("../input/shot_logs.csv")




df.columns




df['SHOT_RESULT'].value_counts()




57905/(70164.+57905.)









df = df.set_index(['player_name','MATCHUP','SHOT_NUMBER']).sort_index()['SHOT_RESULT'].unstack()




df.index = df.index.droplevel()
df.fillna(value=np.nan, inplace=True)




df.head()




from operator import add

for jj in range(1,8):# number of consecutive shots 
    total = 0
    made = 0
    for ii in range(1,38-jj+1):# shots taken (for a player in a game)
        total += (df[list(map(add, [ii-1]*(jj+1), range(1,(jj+1))))]=='made').all(axis=1).sum()
        made +=  (df[list(map(add, [ii-1]*(jj+1), range(1,(jj+2))))]=='made').all(axis=1).sum()
    print('---n = ',jj)
    print('make percentage: ',made/float(total))
    print('n shots were consecutively: ',total)

















































































