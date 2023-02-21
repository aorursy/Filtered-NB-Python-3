#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)




reviews.head()




#check_q1(pd.DataFrame())
#print(answer_q1())




reviews.loc[:,'description']
#reviews.description




reviews.loc[0,'description']
#print(answer_q2())
#check_q2(pd.DataFrame())




reviews.iloc[0,:]
#reviews.iloc[0]




reviews.loc[0:10,'description']




Select the first 10 values from the description column of reviews




reviews.iloc[[1,2,3,5,8],:]




reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]




reviews.loc[0:100,['country','variety']]




reviews.loc[reviews['country']=='Italy',:]




reviews.loc[reviews['region_2'].isnull()==False,:]




reviews.loc[:,'points']




reviews.loc[0:1000,'points']




#reviews.loc[len(reviews):1000:-1,'points']
reviews.iloc[-1000:,3]




reviews.loc[reviews['country']=='Italy','points']




#reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country
reviews.loc[(reviews['country'].isin(["Italy", "France"])) & (reviews['points'] >= 90),'country']

