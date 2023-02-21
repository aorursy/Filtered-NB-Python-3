#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (10, 6)

import matplotlib.pyplot as plt
plt.style.use('ggplot')


df_cacao = pd.read_csv('../input/flavors_of_cacao.csv')

df_cacao.columns




df_cacao.columns = ['company', 'origin_or_name', 'ref', 'review_date', 'cocoa_percent', 'company_location', 'rating', 'bean_type', 'broad_bean_origin']




df_cacao.head()




df_cacao.dtypes




df_cacao.cocoa_percent = df_cacao.cocoa_percent.str.replace('%', '').astype(float)




df_ten_companies = df_cacao.groupby('company')['company'].count()         .sort_values(ascending=False)[:10]         .to_frame()

df_ten_companies.columns = ['Count']

df_ten_companies




df_ten_companies.plot.barh(title='Top 10 companies')




df_ten_countries = df_cacao.groupby('company_location')['company_location']                             .count()                             .sort_values(ascending=False)[:10]                             .to_frame()
df_ten_countries.columns = ['Count']
df_ten_countries




df_ten_countries.plot.barh(title='Top 10 countries')




df_cocoa_percentage = df_cacao.groupby('cocoa_percent')                                 .count()['ref']                                 .sort_values(ascending=False)                                 .reset_index()[:15]
df_cocoa_percentage.columns = ['Cocoa percentage', 'Number of chocolate']
            
df_cocoa_percentage.plot.bar(x='Cocoa percentage', 
                            title='Distribution of chocolate over their cocoa percentage')




location = df_ten_countries.index
df_vio = [df_cacao[df_cacao['company_location'] == loc]['cocoa_percent'] for loc in location]

fig, axes = plt.subplots()

axes.violinplot(df_vio, showmeans=True)
axes.set_xticks(np.arange(1, len(location) + 1))
axes.set_xticklabels(location)
axes.set_title('Cocoa % distribution of the 10 first countries')




df_best_beans = df_cacao.groupby('broad_bean_origin')['rating']                         .aggregate(['mean', 'var', 'count'])                         .replace(np.NaN, 0)                         .sort_values(['mean', 'var'], ascending=[False, False])
df_best_beans.head()




df_best_beans['count'].mean()




df_best_beans = df_best_beans.sort_values('count', ascending=False)[:20]                             .sort_values('mean', ascending=False)
df_best_beans.head()




df_best_beans['mean'].plot.bar(yerr=df_best_beans['var'], title="Places with high rating beans")




df_highest = df_cacao.groupby('company_location')['rating']                         .aggregate(['mean', 'var', 'count'])                         .replace(np.NaN, 0)                         .sort_values(['mean', 'var'], ascending=[False, False])
df_highest.head()




df_highest = df_highest.sort_values('count', ascending=False)[:20]             .sort_values('mean', ascending=False)
    
df_highest.head()




df_highest.plot.bar(y='mean', yerr='var')




location = df_highest.index

df_vio = [df_cacao[df_cacao['company_location'] == loc]['rating'] for loc in location]
fig, axes = plt.subplots(figsize=(14, 10))

axes.violinplot(df_vio, showmeans=True)
axes.set_xticks(np.arange(1, len(location) + 1))
axes.set_xticklabels(location)
axes.xaxis.set_tick_params(rotation=45)
axes.set_title('Rating distribution of the 20 first countries')

