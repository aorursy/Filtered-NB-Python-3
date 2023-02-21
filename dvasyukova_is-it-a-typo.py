#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm




dn = pd.read_csv('../input/NationalNames.csv')
ds = pd.read_csv('../input/StateNames.csv')




dn[dn.Name=='Christop']




ds[ds.Name=='Christop']




import Levenshtein
Levenshtein.distance('Christop','Christopher')




names = dn.groupby(['Name','Gender'])['Count'].sum().reset_index()
names['Distance'] = names.Name.apply(lambda x: Levenshtein.distance(x,'Christop'))




names.loc[(names.Gender=='M')&(names.Distance<=3)]     .sort_values(by=['Count','Distance'], ascending=[False,True]).head(10)




data = ds[(ds.State=='NY')&(ds.Name=='Christopher')&(ds.Gender=='M')]
fig, ax = plt.subplots()
ax.plot(data.Year, data.Count)
c = data.loc[data.Year==1989,'Count'].values[0]
ax.vlines(1989, c, c+1082)
ax.set_xlim(1970,2014);
ax.set_title('Boys named Christopher in NY')
ax.arrow(1985,2500,4,1300)
ax.text(1985,2200,'Boys named Cristop this year',
        horizontalalignment='center')




names = dn.groupby(['Name','Gender'])['Count']          .agg(['sum','count'])          .rename(columns={'sum':'Count','count':'YearsActive'})
names['StatesActive'] = ds.groupby(['Name','Gender'])['State']                          .apply(pd.Series.nunique)
names = names.sort_values(by=['YearsActive','StatesActive','Count'],ascending=[True,True,False])




typos = names[(names.YearsActive==1)&(names.StatesActive==1)]
typos = typos.merge(ds[['Name','Gender','Year','State']], how='left', 
                    left_index=True, right_on=['Name','Gender'])
typos.head()




typos.groupby(['State','Year'])['Name'].count().sort_values(ascending=False).head()




ny = typos.loc[(typos.State=='NY')&(typos.Year==1989),['Count','Name','Gender']]
ny.head(10)




print('total babies affected: {}'.format(ny.Count.sum()))




names = names.reset_index()




def find_full_name(typoname, gender):
    data = names.loc[(names.Gender==gender)&(names.Name.str.startswith(typoname))]
    return data.loc[data.Count.idxmax(),'Name']
find_full_name('Alexandr','F')




ny['FullName'] = ''
for i in ny.index:
    ny.loc[i, 'FullName'] = find_full_name(ny.loc[i,'Name'],ny.loc[i,'Gender'])
ny.head(10)




# under construction




ds[(ds.Name=='Alexandr')&(ds.Gender=='F')]




#names = names.reset_index()
names['Distance'] = names.Name.apply((lambda x: Levenshtein.distance(x,'Alexandr')))
names.loc[(names.Gender=='F')&(names.Distance<=3)]     .sort_values(by=['Count','Distance'], ascending=[False,True]).head(10)




data = ds[(ds.State=='NY')&(ds.Name=='Alexandra')&(ds.Gender=='F')]
fig, ax = plt.subplots()
ax.plot(data.Year, data.Count)
c = data.loc[data.Year==1989,'Count'].values[0]
ax.vlines(1989, c, c+301)
ax.set_xlim(1970,2014);
ax.set_title('Girls named Alexandra in NY')
ax.arrow(1995,400,-6,300)
ax.text(1995,350,'Girls named Alexandr this year',
        horizontalalignment='center')




ds[(ds.Name=='Dalary')&(ds.Gender=='F')]




ds[(ds.Name=='Jacquely')&(ds.Gender=='F')]




names['Distance'] = names.Name.apply((lambda x: Levenshtein.distance(x,'Jacquely')))
names.loc[(names.Gender=='F')&(names.Distance<=3)]     .sort_values(by=['Count','Distance'], ascending=[False,True]).head(10)




data = ds[(ds.State=='NY')&(ds.Name=='Jacquelyn')&(ds.Gender=='F')]
fig, ax = plt.subplots()
ax.plot(data.Year, data.Count)
c = data.loc[data.Year==1989,'Count'].values[0]
ax.vlines(1989, c, c+50)
ax.set_xlim(1970,2014);
ax.set_title('Girls named Jacquelyn in NY')
ax.arrow(1985,40,4,40)
ax.text(1985,35,'Girls named Jacquely this year',
        horizontalalignment='center')




data = ds[(ds.State=='NY')&(ds.Name=='Cassandra')&(ds.Gender=='F')]
fig, ax = plt.subplots()
ax.plot(data.Year, data.Count)
c = data.loc[data.Year==1989,'Count'].values[0]
ax.vlines(1989, c, c+152)
ax.set_xlim(1970,2014);
ax.set_title('Girls named Cassandra in NY')
ax.arrow(1995,200,-6,200)
ax.text(1995,180,'Girls named Cassandr this year',
        horizontalalignment='center')






