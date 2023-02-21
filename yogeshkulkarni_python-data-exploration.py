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




# Next, load the train and test datasets available in the "../input/" directory
train = pd.read_csv("../input/train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("../input/test.csv") # the train dataset is now a Pandas DataFrame

# Let's have a peek of the train data
train.head()




instance_count, attr_count = train.shape
print('Number of instances: ', instance_count)
print('Number of features:', attr_count)




# View the columns
train.columns




# some statistical overview

train.describe()




# Check for missing values
pd.isnull(train).any()




# Count missing values in training data set
pd.isnull(train).sum()




train.mean()




train.fillna(train.mean())




pearson = train.corr(method='pearson')
pearson




# Since the target attr is the last, remove corr with itself
corr_with_target = pearson.iloc[-1,:-1]#pearson.ix[-1][:-1]

corr_with_target_dict = corr_with_target.to_dict()

# List the attributes sorted from the most predictive by their correlation with Sale Price
print("FEATURE \tCORRELATION")
for attr in sorted(corr_with_target_dict.items(), key = lambda x: -abs(x[1])):
    print("{0}: \t{1}".format(*attr))




corr_with_target[abs(corr_with_target).argsort()[::1]]




attrs = pearson.iloc[:-1,:-1] # all except target
# only important correlations and not auto-correlations
threshold = 0.5
# {(YearBuilt, YearRemodAdd): 0.592855, (1stFlrSF, GrLivArea): 0.566024, ...
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0])     .unstack().dropna().to_dict()
#     attribute pair                   correlation
# 0     (OverallQual, TotalBsmtSF)     0.537808
# 1     (GarageArea, GarageCars)	   0.882475
# ...
unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['Attribute Pair', 'Correlation'])
# sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

unique_important_corrs




import seaborn as sns
import matplotlib.pyplot as plt


# Generate a mask for the upper triangle
mask = np.zeros_like(pearson, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(pearson, mask=mask, cmap=cmap, vmax=.3,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)




target = train['SalePrice']
plt.hist(target, bins=50)




sns.distplot(target)




# Scatter Plot
x, y = train['YearBuilt'], train['SalePrice']
plt.scatter(x, y, alpha=0.5)

# or via jointplot (with histograms aside):
sns.jointplot(x, y, kind='scatter', joint_kws={'alpha':0.5})




# Hexagonal 2-D plot
sns.jointplot(x, y, kind='hex')




sns.kdeplot(x, y, shade=True)
# or 
sns.jointplot(x, y, kind='kde')




plt.figure(1)
f, axarr = plt.subplots(3, 2, figsize=(10, 9))
y = target.values
axarr[0, 0].scatter(train['OverallQual'].values, y)
axarr[0, 0].set_title('OverallQual')
axarr[0, 1].scatter(train['TotRmsAbvGrd'].values, y)
axarr[0, 1].set_title('TotRmsAbvGrd')
axarr[1, 0].scatter(train['GarageCars'].values, y)
axarr[1, 0].set_title('GarageCars')
axarr[1, 1].scatter(train['GarageArea'].values, y)
axarr[1, 1].set_title('GarageArea')
axarr[2, 0].scatter(train['TotalBsmtSF'].values, y)
axarr[2, 0].set_title('TotalBsmtSF')
axarr[2, 1].scatter(train['1stFlrSF'].values, y)
axarr[2, 1].set_title('1stFlrSF')
f.text(-0.01, 0.5, 'Sale Price', va='center', rotation='vertical', fontsize = 12)
plt.tight_layout()
plt.show()

