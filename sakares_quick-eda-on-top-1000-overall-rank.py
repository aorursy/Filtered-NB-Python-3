#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.




df = pd.read_csv('../input/marathon_results_2016.csv')

all_runners = df
top1000_runners = df[0:1000]

runners = top1000_runners




all_runners.describe()




ax = sns.boxplot(x="M/F", y="Overall", data=runners)
ax = sns.stripplot(x="M/F", y="Overall", data=runners, jitter=True, edgecolor="gray")




sns.jointplot(x="Age", y="Overall", data=runners, size=7)




sns.FacetGrid(runners, hue="M/F", size=7)   .map(plt.scatter, "Overall", "Age")   .add_legend()




# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot

sns.violinplot(x="Age", y="Overall", data=runners, size=6)




# Helper function
# Credit: https://www.kaggle.com/drgilermo
# Ref: https://www.kaggle.com/drgilermo/d/rojour/boston-results/negative-split-and-the-wall

def time_to_min(string):
    if string is not '-':
        time_segments = string.split(':')
        hours = int(time_segments[0])
        mins = int(time_segments[1])
        sec = int(time_segments[2])
        time = hours*60 + mins + np.true_divide(sec,60)
        return time
    else:
        return -1

print(time_to_min(df.loc[1,'Pace']))




pace_to_min = runners['Pace'].apply(lambda x: time_to_min(x))
pace_to_min.hist(bins=50)




corr = runners.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()
plt.figure(figsize=(7, 7))
sns.heatmap(corr, vmax=1, square=True)

