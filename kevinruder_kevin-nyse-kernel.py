#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




prices = pd.read_csv("../input/prices.csv")

securities = pd.read_csv("../input/securities.csv")

priceSplit = pd.read_csv("../input/prices-split-adjusted.csv")

fundamentals = pd.read_csv("../input/fundamentals.csv")




prices.head(5)




prices.describe()




fundamentals.head(5)




fundamentals.describe()




priceSplit.head()




priceSplit.describe()




prices['date'] = pd.to_datetime(prices['date'])




prices.set_index('date')




def f(x):
    d = []
    d.append(x['open'].diff())
    d.append(x['close'].diff())
    return pd.Series(d,index=['diff_open','diff_close'])

kak = prices.set_index('date').groupby('symbol').max()




prices = prices.set_index(['symbol','date'])




prices




idx = prices.index.get_level_values

idx(0)




diff_pr_symbol = prices.groupby([idx(0)]).apply(lambda df: df - df.shift(1))




overall_diff_pr_year = (diff_pr_symbol.groupby([idx(0)]
                         +[pd.Grouper(freq='Y',level=-1)]).sum())




#largest gains in a year based on opening price

overall_diff_pr_year.groupby('symbol').apply(sum)['open'].sort_values(ascending=False).head(5)




#Biggest losers

overall_diff_pr_year.groupby('symbol').apply(sum)['open'].sort_values(ascending=False).tail(5)




#having a look at the top dog

overall_diff_pr_year.loc['PCLN']




normalized_diff_pr_symbol = diff_pr_symbol.groupby([idx(0)]).apply(lambda df : (df - df.mean())/(df.max() - df.min()))




percent_change_pr_symbol = diff_pr_symbol['open'] / prices.groupby(['symbol','date'])['open'].mean()




percent_change_pr_symbol.head()




percent_change_pr_symbol.groupby('symbol').sum().sort_values(ascending=True).head()









#percentage_change_diff_pr_symbol= diff_pr_symbol.groupby([idx(0)]).apply(lambda df : (df / prices))




norm_overall_diff_pr_year = (normalized_diff_pr_symbol.groupby([idx(0)]
                         +[pd.Grouper(freq='Y',level=-1)]).sum())




#largest gains in a year based on opening price

norm_overall_diff_pr_year.groupby('symbol').apply(sum)['open'].sort_values(ascending=False).head(5)




for company in norm_overall_diff_pr_year.groupby('symbol').apply(sum)['open'].sort_values(ascending=False).head(5).index.values:
    fig = prices.loc[company]['open'].plot.line().legend(norm_overall_diff_pr_year.groupby('symbol').apply(sum)['open'].sort_values(ascending=False).head(5).index.values)
  




for company in norm_overall_diff_pr_year.groupby('symbol').apply(sum)['open'].sort_values(ascending=False).tail(5).index.values:
    fig = prices.loc[company]['open'].plot.line().legend(norm_overall_diff_pr_year.groupby('symbol').apply(sum)['open'].sort_values(ascending=False).tail(5).index.values)




#prices.drop(['symbol','date'],axis=1) - prices.drop(['symbol','date'],axis=1).shift(-1)




import seaborn as sns

#prices[['open','close','low','high','volume']] = prices[['open','close','low','high','volume']].diff()

ax = sns.heatmap(prices.corr(),linecolor='black')

prices.corr()




shortest_time_on_SE = prices.groupby(['symbol','date']).nunique().groupby(['symbol']).sum().low.sort_values().head(20).index.values




norm_overall_diff_pr_year.loc[shortest_time_on_SE].groupby('symbol').sum().open.sort_values(ascending=False)




normalized_diff_pr_symbol.groupby(['symbol','date']).sum().groupby(['symbol']).max().head()




min_max_prices = prices.groupby('symbol').agg({'low': 'min',
                                                'high' :'max'})




min_max_prices['largest_potential_gain_pct'] = (min_max_prices.high - min_max_prices.low) / min_max_prices.low




min_max_prices.head()




#plotting stock opening prices for the largest 5 potential gains.

for company in min_max_prices['largest_potential_gain_pct'].sort_values().tail(5).index.values:
    prices.loc[company].open.plot.line().legend(min_max_prices['largest_potential_gain_pct'].sort_values().tail(5).index.values)




#removed REGN to get a better look at the remaining four companies opening price

for company in min_max_prices['largest_potential_gain_pct'].sort_values().tail(5).drop('REGN').index.values:
    prices.loc[company].open.plot.line().legend(min_max_prices['largest_potential_gain_pct'].sort_values().tail(5).drop('REGN').index.values)




#as percent change - makes it easy to detect dips and increase in price

for company in min_max_prices['largest_potential_gain_pct'].sort_values().tail(5).index.values:
    percent_change_pr_symbol.loc[company].plot.line().legend(min_max_prices['largest_potential_gain_pct'].sort_values().tail(5).index.values)




for x in prices:
        prices['pct_change_'+x+'_diff'] = prices[x].groupby('symbol').pct_change() #this one doesn't seem quite right
        prices[x+'_diff'] = prices[x].groupby('symbol').apply(lambda s: (s - s.shift(1)))
        prices['log_'+x+'_diff'] = prices[x].groupby('symbol').apply(lambda s: np.log(s/s.shift(1)))
    
prices.loc['GOOG'].head(10)




prices.loc['NVDA']




np.log(prices.loc['NVDA'].iloc[-1,:].close) - np.log(prices.loc['NVDA'].iloc[1,:].close)




#calculating stocks with the best return based on the natural log

prices.groupby('symbol').apply(lambda df: np.log(df.iloc[-1,:].close)- np.log(df.iloc[1,:].close)).sort_values(ascending=False)




from scipy import signal
import matplotlib.pyplot as plt

fs = 100

f, Pxx_den = signal.periodogram(prices.loc['GOOG'].close, fs)
plt.semilogy(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()




from sklearn.linear_model import LinearRegression




lrm = LinearRegression()




lrm.fit(X=prices.loc['NVDA'].close.index.values.astype(int).reshape(-1,1),y=prices.loc['NVDA'].close)




predictions = lrm.predict(prices.loc['NVDA'].close.index.values.astype(int).reshape(-1,1))




Nvda = prices.loc['NVDA']




Nvda['pred'] = predictions




simeplegraph = Nvda[['pred','close']]




#KEEP IN MIND THESE ARE ONLY USED AS A FEATURE FOR THE PREDICTION MODEL AND ARE NOT THE FINAL PREDICTION




simeplegraph.plot.line()




def apply_lr(df):
    lrm = LinearRegression()
    lrm.fit(X=np.array(df.close.index.get_level_values(1).astype(int)).reshape(-1,1),y=df.close)
    prediction = lrm.predict(np.array(df.close.index.get_level_values(1).astype(int)).reshape(-1,1))
    df['close_lr_pred'] = prediction
    return df
    




def apply_lr_diff(df):
    df['diff_pred_to_true'] = df.close - df.close_lr_pred
    return df




prices = prices.groupby('symbol').apply(apply_lr)




#Simple visualization of our regression




for company in min_max_prices['largest_potential_gain_pct'].sort_values().head(5).index.values:
    prices.loc[company].close.plot.line()
    prices.loc[company].close_lr_pred.plot.line()
    




prices = prices.groupby('symbol').apply(apply_lr_diff)




prices









def calc_pvr(df):
    
    #making a temporary data frame for each iteration of variance
    
    short_variance = pd.DataFrame()
    long_variance = pd.DataFrame()
    
   #for the variance over a period of 7 days, we make 7 columns, each column is the price shifted i-times (range 0,7)
    for i in range(0,7):
        short_variance['{}'.format(i)] = df.close.shift(i)
    
    #for the variance over a period of 60 days, we make 60 columns, each column is the price shifted i-times (range 0,60)
    for i in range(0,60):
        long_variance['{}'.format(i)] = df.close.shift(i)
    
    #fill in the Na fields by using ffil method along the rows
    
    short_variance = short_variance.fillna(method='ffill',axis=1)
    
    long_variance = long_variance.fillna(method='ffill',axis=1)
    
    #calculate the mean across the rows
    
    mean_sv = short_variance.mean(axis=1)
    
    mean_lv = long_variance.mean(axis=1)
    
    #res and res2 are the calculated variance of each dataframe
    
    res = short_variance.apply(lambda df: np.square(df-mean_sv)).apply(sum,axis=1).apply(lambda df : df/(len(short_variance.columns)-1))
    
    res2 = long_variance.apply(lambda df: np.square(df-mean_lv)).apply(sum,axis=1).apply(lambda df : df/(len(long_variance.columns)-1))
    
    pvr = res/res2

    df['pvr'] = pvr
    
    return df




prices = prices.groupby('symbol').apply(calc_pvr)




#Pretty interesting if you look up certain dates you will be able to see that the dates correlates to events of imprortance f.ex. the launch of google nexus correspondes to the largest PVR value




prices.loc['GOOG'].pvr.sort_values(ascending=False).head(10)




#you also see some periodicity in regards to what I would assume are releases of quarterly and yearly reports




prices.loc['GOOG'].pvr.plot.line()




prices_ss = prices.sample(frac= 0.10)




def calc_ATR(df):
    
    atr_df = pd.DataFrame()
    
    atr_df['TR'] = df.high - df.low
        
    #calculating the average true range over a period of 14 days
    for i in range(1,13):
        atr_df['TR{}'.format(i)] = atr_df.TR.shift(i)        
    
    atr_df = atr_df.fillna(method='ffill',axis=1)
    prev_atr = atr_df.iloc[:,1:13].mean(axis=1)
    df['ATR'] = ((prev_atr * 13) + atr_df.TR)/14
    
    
    return df
    
        
    
    




prices = prices.groupby('symbol').apply(calc_ATR)




prices.head(10)




def calc_MMI(df):
    #Market meanness index. An indication of whether or not the trend is about to change
    MMI_df = pd.DataFrame()
    df['MMI'] = None
    df_uidx = df.reset_index()    
    #calculating the average true range over a period of 14 days
    for i in range(13,len(df)):
        medi = df_uidx.close.loc[i-13:i].median()    
        df_uidx.MMI.loc[i] = medi
        nh = 0
        nl = 0
        for k in range(0,13):
            if df_uidx.close.loc[i-k] > medi:
                nh +=1
            else:
                nl += 1
                
        df_uidx.MMI.loc[i] = 100*(nh+nl)/13
    
    return  df_uidx
    
        




#prices_ss.groupby('symbol').apply(calc_MMI).loc['GOOG']




def calc_MMI2(df):
    #Market meanness index. An indication of whether or not the trend is about to change
    MMI_df = pd.DataFrame()
    
    for i in range(0,13):
        MMI_df['{}'.format(i)] = df.close.shift(i)
    
    MMI_df.fillna(method='ffill',axis=1)
    
    medi = MMI_df.median(axis=1)
    
    nh = MMI_df.apply(lambda df: ((df > medi) & (df > df.shift(1))))
    
    nl = MMI_df.apply(lambda df: ((df < medi) & (df < df.shift(1))))
    
    MMI = pd.concat(objs=(nl,nh),axis=1).sum(axis=1)  # .reset_index().drop('symbol',axis=1).set_index('date')
    
    MMI = (100 * MMI / 13)
    
    df['MMI'] = MMI
    
    return df
        
        
    




MMI = prices.groupby('symbol').apply(calc_MMI2)




prices_ss = prices_ss.sample(frac=0.10)




MMI




def calc_MMI_deviation(df):
    #Market meanness index. An indication of whether or not the trend is about to change
    
    zero = pd.DataFrame(data=[df.iloc[0]])
    
    shifted = df.drop(df.MMI.index[0]).shift(7).fillna(method='pad')

    shifted_MMI = pd.concat([zero,shifted],axis=0)
    
    df['MMI_d'] = (df.MMI/shifted_MMI.MMI.fillna(method='bfill'))
    
    return df
        
        
    









#prices_ss.groupby('symbol').apply(calc_MMI_deviation).loc['GOOG']




prices = MMI.groupby('symbol').apply(calc_MMI_deviation)




prices




sample_company = prices.loc['GOOG'][['close','pct_change_volume_diff','close_lr_pred','pvr','ATR','MMI','MMI_d','log_close_diff']]




ax = sns.heatmap(sample_company.corr(),linecolor='black')




from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(sample_company.dropna().shift(1).fillna(method='bfill'), sample_company.dropna().log_close_diff, test_size=0.1,shuffle=False)




#svr_poly.fit(sample_company.drop('close',axis=1).dropna(),sample_company.dropna().close)




lrm = LinearRegression()




lrm.fit(X_train,y_train)




a = lrm.predict(X_test)




pd.Series(a,index=y_test.index).plot.line()
y_test.plot.line()










pd.Series(lrm.coef_,index=sample_company.columns)




from sklearn.metrics import mean_squared_error




mean_squared_error(a,y_test)




from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=1000, random_state=0)




regr.fit(X_train,y_train)




pd.Series(regr.feature_importances_,index=X_train.columns)




b = regr.predict(X_test)




pd.Series(b,index=y_test.index).plot.line()
y_test.plot.line()









mean_squared_error(b,y_test)









import statsmodels.api as sm




#Holt’s Linear Trend method

sm.tsa.seasonal_decompose(X_test.log_close_diff, freq=30).plot()
result = sm.tsa.stattools.adfuller(X_test.close)
plt.show()




import math
def savings(inp,returns,months):
    collective = 0 + inp
    for month in range(months) :
        collective += (inp)
        collective *= math.exp(returns)
    return collective




savings(10000,0.05,24)






