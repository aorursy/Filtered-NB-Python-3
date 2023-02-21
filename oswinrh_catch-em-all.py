#!/usr/bin/env python
# coding: utf-8



import numpy as np # numpy is a library for linear algebra
import pandas as pd # pandas is for data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # for generating pseudo-random numbers
import datetime, pytz # for manipulating dates and times
import io # provides the Python interfaces to stream handling
import requests # allows you to send organic, grass-fed HTTP/1.1 requests
import seaborn as sb # visualization library based on matplotlib
import matplotlib as mpl # famous 2D plotting library
import matplotlib.pyplot as plp 
import sklearn # implements machine learning, preprocessing, cross-validation and visualization algorithms
import sqlite3 # performs SQL on Python




poke = pd.read_csv('../input/Pokemon.csv')




poke.head()




poke.describe()




poke["Type 1"].value_counts()




types = poke['Type 1']
colors = ['turquoise','white','lightgreen','green','purple','red','chocolate','yellow','brown','yellowgreen'
          ,'lavender','grey','lightcoral','darkgrey','silver','lightblue','pink','orange']
explode = np.arange(len(types.unique())) * 0.01

types.value_counts().plot.pie(
    explode=explode,
    colors=colors,
    title="Percentage of Different Types of Pokemon",
    autopct='%1.1f%%',
    shadow=True,
    startangle=90,
    figsize=(8,8)
)
plp.tight_layout()




sb.FacetGrid(poke, hue="Legendary", size=8)    .map(plp.scatter, "Defense", "Attack")    .add_legend()




typehp = poke[['Type 1', 'HP']].groupby(['Type 1'], as_index=False).mean().sort_values(by='HP', ascending=False)
sb.barplot(x='Type 1', y='HP', data=typehp)




c = sqlite3.connect(':memory:')
pd.read_csv('../input/Pokemon.csv').to_sql('poke',c)




pd.read_sql("SELECT Name, `Type 1`, max(HP) FROM poke", c)




pd.read_sql("SELECT Name, `Type 1`, min(HP) FROM poke", c)




pd.read_sql("SELECT Name, [Type 1], [Type 2], Total, HP, Attack, Defense, Speed, [Sp. Atk], [Sp. Def], Legendary, Generation FROM poke WHERE Legendary = '1' ORDER BY 4 DESC LIMIT 5", c)




pd.read_sql("SELECT Name, [Type 1], [Type 2], Total, HP, Attack, Defense, Speed, [Sp. Atk], [Sp. Def], Legendary, Generation FROM poke WHERE Legendary = '0' ORDER BY 4 DESC LIMIT 5", c)




pd.read_sql("SELECT Name, [Type 1], [Type 2], Total, HP, Attack, Defense, Speed, [Sp. Atk], [Sp. Def], Legendary, Generation FROM poke WHERE Legendary = '1' ORDER BY 5 DESC LIMIT 5", c)




pd.read_sql("SELECT Name, [Type 1], [Type 2], Total, HP, Attack, Defense, Speed, [Sp. Atk], [Sp. Def], Legendary, Generation FROM poke WHERE Legendary = '0' ORDER BY 5 DESC LIMIT 5", c)




pd.read_sql("SELECT [Type 1], avg(HP), avg(Attack), avg(Defense), avg(Speed), max(HP), min(HP), count(Name) FROM poke GROUP BY 1 ORDER BY 2 DESC", c)




pd.read_sql("SELECT Generation, count(Name) FROM poke GROUP BY 1", c)






