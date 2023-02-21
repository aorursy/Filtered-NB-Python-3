#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# TODO need to change to downloadable URL format
# 
dirname = '/kaggle/input'
filename = 'data.csv'
filepath = os.path.join(dirname, filename)




df = pd.read_csv(filepath)




df.head()




df.tail()




# Unique names either gender.
df["name"].nunique()

# Unique names for male.
df[df["gender"] == "M"]["name"].nunique()

# Unique names for female.
df[df["gender"] == "F"]["name"].nunique()

# Unique names for gender neutral.
both_df = df.pivot_table(index="name", columns="gender", values="count", aggfunc=np.sum).dropna()
both_df.index.nunique()




# Step by step approach, the one-liners can be found below their respective tables.
only_gender_male = df[df["gender"] == "M"]
only_name_and_count_colmns = only_gender_male[["name", "count"]]
df_group_by_name = only_name_and_count_colmns.groupby("name")
df_group_by_name_sum = df_group_by_name.sum()
df_group_by_name_sum_sort_by_count = df_group_by_name_sum.sort_values("count", ascending=False)
df_group_by_name_sum_sort_by_count.head(10)




# In one liner format 
df[df["gender"] == "M"][["name", "count"]].groupby("name").sum().sort_values("count", ascending=False).head(10)




# One liner format for Female children
 df[df["gender"] == "F"][["name", "count"]].groupby("name").sum().sort_values("count", ascending=False).head(10)




df_pvt = df.pivot_table(index="name", columns="gender", values="count", aggfunc=np.sum).dropna()




df_pvt_count_gt_50k = df_pvt[(df_pvt["M"] >= 50000) & (df_pvt["F"] >= 50000)]
df_pvt_count_gt_50k.head(20)




both_df = df.groupby("year").sum()
male_df = df[df["gender"] == "M"].groupby("year").sum()
female_df = df[df["gender"] == "F"].groupby("year").sum()

# Initializing list
data = []

# Combined Min (count and year)
both_df_min = both_df.min()["count"]
both_df_count = both_df.idxmin()["count"]

# Appending result to list
data.append(['Both Min',both_df_min, both_df_count ])

# Male Min (count and year)
male_df_min = male_df.min()["count"]
male_df_count = male_df.idxmin()["count"]

# Appending result to list
data.append(['Male Min',male_df_min, male_df_count ])

# Female Min (count and year)
female_df_min = female_df.min()["count"]
female_df_count = female_df.idxmin()["count"]

# Appending to list
data.append(['Female Min',female_df_min, female_df_count ])

# Combined Max (count and year)
both_df_max = both_df.max()["count"]
both_df_max_count = both_df.idxmax()["count"]

# Appending result to list
data.append(['Both Max',both_df_max, both_df_max_count ])

# Male Max (count and year)
male_df_max = male_df.max()["count"]
male_df_max_count = male_df.idxmax()["count"]

# Appending result to list
data.append(['Male Max',male_df_max, male_df_max_count ])

# Female Max (count and year)
female_df_max = female_df.max()["count"]
female_df_max_count = female_df.idxmax()["count"]

# Appending to list final value
data.append(['Female Max',female_df_max, female_df_max_count ])




pd.DataFrame(data, columns=["Gender and Attribute", "Total Count", "Year"])




# Those parameters generate plots with a mauve color.
sns.set(style="ticks",
        rc={
            "figure.figsize": [12, 7],
            "text.color": "white",
            "axes.labelcolor": "white",
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "axes.facecolor": "#443941",
            "figure.facecolor": "#443941"}
        )




both_df = df.groupby("year").sum()
male_df = df[df["gender"] == "M"].groupby("year").sum()
female_df = df[df["gender"] == "F"].groupby("year").sum()




plt.plot(both_df, label="Both", color="yellow")
plt.plot(male_df, label="Male", color="lightblue")
plt.plot(female_df, label="Female", color="pink")




pivoted_df = df.pivot_table(index="name", columns="year", values="count", aggfunc=np.sum).fillna(0)




percentage_df = pivoted_df / pivoted_df.sum() * 100




percentage_df["total"] = percentage_df.sum(axis=1)




sorted_df = percentage_df.sort_values(by="total", ascending=False).drop("total", axis=1)[0:10]




transposed_df = sorted_df.transpose()




transposed_df.columns.tolist()




for name in transposed_df.columns.tolist():
    plt.plot(transposed_df.index, pivoted_df[name], label=name)




yticks_labels = ["{}%".format(i) for i in np.arange(0, 5.5, 0.5)]
plt.yticks(np.arange(0, 5.5, 0.5), yticks_labels)




plt.legend()
plt.grid(False)
plt.xlabel("Year")
plt.ylabel("Percentage by Year")
plt.title("Top 10 Names Growth")
plt.show()

