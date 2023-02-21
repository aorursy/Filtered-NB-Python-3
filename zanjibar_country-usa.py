#!/usr/bin/env python
# coding: utf-8



# -*- encoding:utf-8 -*-
import pandas.io.sql as psql
import sqlite3
from sklearn import linear_model
from IPython.display import display, HTML
from datetime import datetime as dt
import time
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import re
import json
import inspect
import requests
import codecs
import platform
import numpy as np
import pandas as pd
from matplotlib import ticker
get_ipython().run_line_magic('matplotlib', 'inline')
clf = linear_model.LinearRegression()
# http://www.customs.go.jp/toukei/info/tsdl_e.htm
show_tables = "select tbl_name from sqlite_master where type = 'table'"
desc = "PRAGMA table_info([{table}])"

conn =     sqlite3.connect(':memory')
cursor = conn.cursor()




# year_from_1997
attach = 'attach "../input/japan-trade-statistics/y_1997.db" as y_1997'
cursor.execute(attach)
# hs code,country,
attach = 'attach "../input/japantradestatistics2/trade_meta_data.db" as code'
cursor.execute(attach)
# import hs,country code as pandas
tmpl = "{hs}_{lang}_df =  pd.read_sql('select * from code.{hs}_{lang}',conn)"
for hs in ['hs2','hs4','hs6','hs6','hs9']:
    for lang in ['jpn','eng']:
        exec(tmpl.format(hs=hs,lang=lang))        
# country 
country_eng_df = pd.read_sql('select * from code.country_eng',conn)
country_eng_df['Country']=country_eng_df['Country'].apply(str)
country_jpn_df = pd.read_sql('select * from code.country_jpn',conn)
country_jpn_df['Country']=country_jpn_df['Country'].apply(str)




sql="""
create table y_country as select Year,exp_imp,Country,sum(Value) as Value from 
y_1997.year_from_1997
group by Year,exp_imp,Country
"""[1:-1]
cursor.execute(sql)
sql="""
create table y_country_{exp_imp} as select Year,Country,Value as {name} from
y_country
where exp_imp={exp_imp}
"""[1:-1]

exp_imp = 1
name = 'export'
cursor.execute(sql.format(exp_imp=exp_imp,name=name))
exp_imp = 2
name = 'import'
cursor.execute(sql.format(exp_imp=exp_imp,name=name))
sql="""
create table y_country_total as 
select y1.Year,y1.Country,y1.Year,y1.export + y2.import as total 
from y_country_1 y1 ,y_country_2 y2 
where y1.Year = y2.Year and y1.Country = y2.Country
"""[1:-1]
cursor.execute(sql)




country = '304' #string
country_name = 'USA'
last_year = 2018
# year_from_1997 group by Country




# Country trade total  ranking 2018
sql = """
select y.Country,c.Country_name,total 
from y_country_total y,code.country_eng c
where Year={last_year} and
y.Country = c.Country
order by total desc
"""[1:-1]

df = pd.read_sql(sql.format(last_year=last_year),conn)
df.head(30)




sql = """
select Year,exp_imp,Value 
from y_country 
where Country='{country}' 
order by Year
"""[1:-1]

df = pd.read_sql(sql.format(country=country),conn)
df.head(30)
plt.figure(figsize=(20, 10))
ax  = sns.lineplot(x='Year',y='Value',hue='exp_imp',linewidth = 7.0,
             palette={1: "b", 2: "r"},
             data=df)
ax.legend_._loc = 2
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))




get_ipython().run_cell_magic('time', '', 'sql_last_year="""\nselect y.hs6,h.hs6_name,sum(Value) as Value \nfrom y_1997.year_from_1997 y,code.hs6_eng h\nwhere Year={last_year} and\nexp_imp={exp_imp} and\nCountry=\'{country}\' and\ny.hs6 !=\'000000\' and\ny.hs6 = h.hs6 \ngroup by y.hs6\norder by Value desc\nlimit {limit}\n"""[1:-1]\nexp_imp = 1\nlimit = 7\ndf = pd.read_sql(sql_last_year.format(last_year=last_year,country=country,limit=limit,exp_imp=exp_imp),conn)\ncodes = \',\'.join([\'"\' + x + \'"\' for x in list(df[\'hs6\'])])\ndf')




get_ipython().run_cell_magic('time', '', 'sql_y = """\nselect Year,y.hs6,h.hs6_name,sum(Value) as Value \nfrom y_1997.year_from_1997 y,code.hs6_eng h\nwhere \nexp_imp={exp_imp} and\nCountry=\'{country}\' and\ny.hs6 !=\'000000\' and\ny.hs6 in ({codes}) and\ny.hs6 = h.hs6 \ngroup by Year,y.hs6\norder by Value desc\n"""[1:-1]\ndf = pd.read_sql(sql_y.format(country=country,codes=codes,exp_imp=exp_imp),conn)\ndf[\'hs6_name\'] = df[\'hs6_name\'] + df[\'hs6\']\n\nplt.figure(figsize=(20, 10))\nsns.lineplot(x=\'Year\',y=\'Value\',hue=\'hs6_name\',linewidth = 7.0,data=df)\nax.xaxis.set_major_locator(ticker.MultipleLocator(1))')




exp_imp = 2
df = pd.read_sql(sql_last_year.format(last_year=last_year,country=country,limit=limit,exp_imp=exp_imp),conn)
codes = ','.join(['"' + x + '"' for x in list(df['hs6'])])
df




df = pd.read_sql(sql_y.format(country=country,codes=codes,exp_imp=exp_imp),conn)
df['hs6_name'] = df['hs6_name'] + df['hs6']

plt.figure(figsize=(20, 10))
sns.lineplot(x='Year',y='Value',hue='hs6_name',linewidth = 7.0,data=df)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

