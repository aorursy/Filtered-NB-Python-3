#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




df_Deaths = pd.read_csv('../input/DeathRecords.csv', usecols=[ "Sex", "Age"])




df_Deaths.pivot_table(index=["Sex"], values=["Age"], aggfunc=np.mean)




df_Deaths.pivot_table(index=["Sex"], values=["Age"],aggfunc=np.mean).     plot(kind='bar', color="blue", grid='on', ylim=(68,78),     title="Average Age of Death by Gender")




df_deaths_m = df_Deaths[df_Deaths.Sex == 'M']
df_dist_m = df_deaths_m[df_deaths_m.Age < 150].groupby(['Age']).Sex.count()
df_deaths_f = df_Deaths[df_Deaths.Sex == 'F']
df_dist_f = df_deaths_f[df_deaths_f.Age < 150].groupby(['Age']).Sex.count()
plt.plot(df_dist_m, color = "red", label="Males")
plt.plot(df_dist_f, color = "green", label="Females")
plt.legend(loc='upper left')
plt.title("2014 Number of Deaths by Gender")






