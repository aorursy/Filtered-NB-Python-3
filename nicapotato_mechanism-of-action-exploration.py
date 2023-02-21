#!/usr/bin/env python
# coding: utf-8



# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import time
import math
import itertools
from itertools import combinations
import re
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

sns.set_style("whitegrid")
notebookstart = time.time()
pd.options.display.max_colwidth = 500
pd.options.display.max_rows = 999
pd.options.display.width = 300
pd.options.display.max_columns = 100




def big_count_plotter(plot_df, plt_set, columns, figsize, hue = None,
                      custom_palette = sns.color_palette("Paired", 15), top_n = 15):
    """
    Iteratively Plot all categorical columns
    Has category pre-processing - remove whitespace, lower, title, and takes first 30 characters.
    """
    rows = math.ceil(len(plt_set)/columns)
    n_plots = rows*columns
    f,ax = plt.subplots(rows, columns, figsize = figsize)
    for i in range(0,n_plots):
        ax = plt.subplot(rows, columns, i+1)
        if i < len(plt_set):
            c_col = plt_set[i]
            plt_tmp = plot_df.loc[plot_df[c_col].notnull(),c_col]                .astype(str).str.lower().str.strip()                .str.title().apply(lambda x: x[:30])
            plot_order = plt_tmp.value_counts().index[:top_n]
            if hue:
                sns.countplot(y = plt_tmp, ax = ax, hue = hue, order = plot_order, palette = custom_palette)
            else:
                sns.countplot(y = plt_tmp, ax = ax, order = plot_order, palette = custom_palette)
            ax.set_title("{} - {} Missing".format(c_col.title(), plot_df[c_col].isnull().sum()))
            ax.set_ylabel("{} Categories".format(c_col.title()))
            ax.set_xlabel("Count")
        else:
            ax.axis('off')

    plt.tight_layout(pad=1)
    
    
def big_boxplotter(plot_df, plt_set, columns, figsize, hue = None, plottype='kde',
                   custom_palette = sns.color_palette("Dark2", 15), quantile = .99):
    rows = math.ceil(len(plt_set)/columns)
    n_plots = rows*columns
    f,ax = plt.subplots(rows, columns, figsize = figsize)
    palette = itertools.cycle(custom_palette)
    for i in range(0,n_plots):
        ax = plt.subplot(rows, columns, i+1)
        if i < len(plt_set):
            cont_col = plt_set[i]
            if hue:
                plt_tmp = plot_df.loc[(plot_df[cont_col].notnull()) & 
                                          (plot_df[cont_col] < plot_df[cont_col].quantile(quantile)),
                                      [cont_col, hue]]
                if plottype == 'box':
                    sns.boxplot(data=plt_tmp, x=cont_col, y=hue, color = next(palette), ax=ax)
                    ax.set_ylabel("Categories")
                elif plottype == 'kde':
                    for h in plt_tmp.dropna()[hue].value_counts()[:5].index:
                        c = next(palette)
                        sns.distplot(plt_tmp.loc[plt_tmp[hue] == h,cont_col], bins=10, kde=True, ax=ax,
                                     kde_kws={"color": c, "lw": 2, "label":h}, color=c)
                    ax.set_ylabel("Density Occurence")
            else:
                plt_tmp = plot_df.loc[(plot_df[cont_col].notnull()) &
                                          (plot_df[cont_col] < plot_df[cont_col].quantile(quantile)),
                                      cont_col].astype(float)
                if plottype == 'box':
                    sns.boxplot(plt_tmp, color = next(palette), ax=ax)
                    ax.set_ylabel("Categories")
                elif plottype == 'kde':
                    sns.distplot(plt_tmp, bins=10, kde=True, ax=ax,
                        kde_kws={"color": "k", "lw": 2}, color=next(palette))
                    ax.set_ylabel("Density Occurence")
            ax.set_title("{} - {:.0f} Missing - {:.2f} Max".format(cont_col.title(),
                plot_df[cont_col].isnull().sum(), plot_df[cont_col].max()))
            ax.set_xlabel("Value")
            
        else:
            ax.axis('off')

    plt.tight_layout(pad=1)
    
    
def rank_correlations(df, figsize=(12,20), n_charts = 18, polyorder = 2, custom_palette = sns.color_palette("Paired", 5)):
    # Rank Correlations
    palette = itertools.cycle(custom_palette)
    continuous_rankedcorr = (df
                             .corr()
                             .unstack()
                             .drop_duplicates().reset_index())
    continuous_rankedcorr.columns = ["f1","f2","Correlation Coefficient"]
    continuous_rankedcorr['abs_cor'] = abs(continuous_rankedcorr["Correlation Coefficient"])
    continuous_rankedcorr.sort_values(by='abs_cor', ascending=False, inplace=True)

    # Plot Top Correlations
    top_corr = [(x,y,cor) for x,y,cor in list(continuous_rankedcorr.iloc[:, :3].values) if x != y]
    f, axes = plt.subplots(int(n_charts/3),3, figsize=figsize, sharex=False, sharey=False)
    row = 0
    col = 0
    for (x,y, cor) in top_corr[:n_charts]:
        if col == 3:
            col = 0
            row += 1
        g = sns.regplot(x=x, y=y, data=df, order=polyorder, ax = axes[row,col], color=next(palette))
        axes[row,col].set_title('{} and {}'.format(x, y))
        axes[row,col].text(0.18, 0.93,"Cor Coef: {:.2f}".format(cor),
                           ha='center', va='center', transform=axes[row,col].transAxes)
        col += 1
    plt.tight_layout(pad=0)
    plt.show()
    
    
# Data Exploration
def custom_describe(df, value_count_n = 5):
    """
    Custom Describe Function - More Tailored to categorical type variables..
    """
    unique_count = []
    for x in df.columns:
        unique_values_count = df[x].nunique()
        value_count = df[x].value_counts().iloc[:5]

        value_count_list = []
        value_count_string = []
        
        for vc_i in range(0,value_count_n):
            value_count_string += ["ValCount {}".format(vc_i+1),
                                   "Occ"]
            if vc_i <= unique_values_count - 1:
                value_count_list.append(value_count.index[vc_i])
                value_count_list.append(value_count.iloc[vc_i])
            else:
                value_count_list.append(np.nan)
                value_count_list.append(np.nan)
        
        unique_count.append([x,
                             unique_values_count,
                             df[x].isnull().sum(),
                             df[x].dtypes] + value_count_list)
        
    print("Dataframe Dimension: {} Rows, {} Columns".format(*df.shape))
    return pd.DataFrame(unique_count,
            columns=["Column","Unique","Missing","dtype"
                    ] + value_count_string
                       ).set_index("Column")

print("Helper Functions Ready")




df = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
print("Train DF Shape: {} Rows, {} Columns".format(*df.shape))

anon_cols = [x for x in df.columns if x not in ['sig_id','cp_type','cp_time','cp_dose']]
categorical_cols = ["cp_dose","cp_type"]
continuous_cols = ["cp_time"] + anon_cols

labels = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
print("Labels DF Shape: {} Rows, {} Columns".format(*labels.shape))

label_names = [x for x in labels.columns if x not in "sig_id"]




melt_labels = pd.melt(labels, id_vars='sig_id', value_vars=label_names)
melt_labels = melt_labels.loc[melt_labels.value != 0].drop("value", axis=1)
melt_labels.rename(columns={"variable":"label"}, inplace=True)

f, ax = plt.subplots(figsize=[10,10])
sns.countplot(y=melt_labels["label"],
              order=melt_labels["label"].value_counts().index[:40],
              palette=sns.color_palette("Paired", 15),
              ax=ax)
ax.set_title("Label Count Plot")
ax.set_ylabel("Labels")
ax.set_xlabel("Counts")
plt.show()




print("Number of active MoA Count:\n{}".format(
    labels.sum(axis=1).value_counts().sort_index().to_dict()))




label_tokens = [item for sublist in [x.split("_") for x in label_names] for item in sublist]
nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
filtered = [w for w in label_tokens if nonPunct.match(w)]
counts = Counter(filtered)
label_keywords_pd = pd.Series(filtered)

f, ax = plt.subplots(figsize=[10,6])
sns.countplot(y=label_keywords_pd,
              order=label_keywords_pd.value_counts().index[:25],
              palette=sns.color_palette("Paired", 15),
              ax=ax
             )
ax.set_title("Label Keywords")
ax.set_ylabel("Label Keywords")
ax.set_xlabel("Counts")
plt.show()




df.sample(5)




big_count_plotter(plot_df = df,
                  plt_set = categorical_cols,
                  columns = 2,
                  figsize = [10,5],
                  custom_palette = sns.color_palette("Paired", 15))




print("Continuous Variables")
display(df[continuous_cols].describe().T.sample(50))




rank_correlations(df = df.loc[:,continuous_cols])




print("Script Complete - Runtime: {:.2f} Minutes".format((time.time() - notebookstart) / 60))






