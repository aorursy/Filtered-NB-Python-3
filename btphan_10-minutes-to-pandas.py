#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




s = pd.Series([1,3,5,np.nan,6,8])
s




dates = pd.date_range('20141101', periods=6)
dates




df = pd.DataFrame(np.random.randn(6,4), index=dates,columns=['one','two','three','four'])
df




df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })

df2




df2.dtypes




df.head()




df.tail(3)




df.index




df.columns




df.values




df.describe()




df.T




df.sort_index(axis=1, ascending=False)




df.sort_values(by='two')




df['one']




df.one




df[0:3]




df['20141102':'20141104']




df.loc[dates[0]]




df.loc[:,['one','two']]




df.loc['20141102':'20141104',['one','two']]




df.loc['20141102',['one','two']]




df.loc[dates[0],'one']




df.at[dates[0],'one']




df.iloc[3]




df.iloc[[1,2,4],[0,2]]




df.iloc[1:3,:]




df.iloc[:,1:3]




df.iloc[1,1]




df.iat[0,0]




df[df.one > 0.5]




df[df>0]




df2 = df.copy()
df2['five'] = ['one', 'one','two','three','four','three']
df2




df2[df2['five'].isin(['two','four'])]




s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20141101', periods=6))
s1




df['six'] = s1
df




df.at[dates[0],'one'] = 0
df




df.iat[0,1] = 0
df




df.loc[:,'four'] = np.array([5] * len(df))
df




df2 = df.copy()
df2[df2 > 0] = -df2
df2




df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
df1




df1.dropna(how='any')




df1.fillna(value=5)




df.mean()




df.mean(1)




s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
s




df.sub(s, axis='index')




df.apply(np.cumsum)




s = pd.Series(np.random.randint(0, 7, size=10))
s




s.value_counts()




s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])




s.str.lower()




df = pd.DataFrame(np.random.randn(10, 4))
df




#break it into pieces:
pieces = [df[:3], df[3:7], df[7:]]
pieces




pd.concat(pieces)




left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
left




right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
right




pd.merge(left,right,on='key')




left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
left




right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
right




pd.merge(left, right, on='key')




df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
df




s = df.iloc[3]
s




df.append(s,ignore_index=True)




df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
df




df.groupby('A').sum()




df.groupby(['A','B']).sum()




tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
tuples




index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
index




df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df




stacked = df.stack()
stacked




stacked.unstack()




stacked.unstack(1)




stacked.unstack(0)




df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})
df




pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])




rng = pd.date_range('1/1/2017', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()




rng = pd.date_range('3/6/2017 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts




ts_utc = ts.tz_localize('UTC')
ts_utc




ts_utc.tz_convert('US/Eastern')




rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts




ps = ts.to_period('M')
ps




ps.to_timestamp()




prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts = pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()




df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df




df["grade"] = df["raw_grade"].astype("category")
df["grade"]




df["grade"].cat.categories = ["very good", "good", "very bad"]




df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df["grade"]

Sorting is per order in the categories, not lexical order.


df.sort_values(by="grade")




df.groupby("grade").size()




ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()




df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')




df.to_csv('foo.csv')




pd.read_csv('foo.csv')




df.to_hdf('foo.h5','df')




pd.read_hdf('foo.h5','df')




df.to_excel('foo.xlsx', sheet_name='Sheet1')




pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])




if pd.Series([False, True, False]):
    print("I was true")

