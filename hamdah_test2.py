#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




get_ipython().run_line_magic('matplotlib', 'inline')




from matplotlib import cm




df = pd.read_csv("../input/database.csv",parse_dates=True)




df.head()




df['Liquid Subtype'].value_counts()




df['Pipeline Type'].value_counts()




df.head()




df['Shutdown Duration'] = (pd.to_datetime(df['Restart Date/Time']) - pd.to_datetime(df['Shutdown Date/Time'])).astype('timedelta64[h]')




df[df['Shutdown Duration'] < 2500]['Shutdown Duration'].hist(bins=50) #.value_counts()




cmap = cm.get_cmap('Reds')
df.plot(x='Shutdown Duration',y='Unintentional Release (Barrels)',kind='scatter')




import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
ax.scatter(x=df['Shutdown Duration'],y=df['Unintentional Release (Barrels)'],c=df['All Costs'],alpha=0.5)
