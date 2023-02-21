#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np




columns = ['PatientAge', 'PatientYear', 'AxillaryNodes', 'SurvivalStatus']
haberman = pd.read_csv('../input/haberman.csv', names = columns)




haberman.head()




haberman.tail()




print(haberman.shape)
print("Haberman Dataset contains {} rows and {} columns".format(*haberman.shape))




print(haberman.columns)




print("Unique values in the SurvivalStatus are {}".format(haberman.SurvivalStatus.unique()))
haberman['SurvivalStatus'].value_counts()




print(haberman.info())
print('*'*50)
print(haberman.describe())

print('*'*50)

for i in range(4):
    print("Class of {} is {}".format(haberman.columns[i],type(haberman.iloc[i][0])))




haberman['SurvivalStatus'] = haberman['SurvivalStatus'].map({1:"yes", 2:"no"})
haberman['SurvivalStatus'] = haberman['SurvivalStatus'].astype('category')
print(haberman.head())
haberman['SurvivalStatus'].value_counts(normalize = True)




for index, column in enumerate(list(haberman.columns)[:-1]):
    fg = sns.FacetGrid(haberman, hue='SurvivalStatus', size=5)
    fg.map(sns.distplot, column).add_legend()
    plt.show()




for index, column in enumerate(list(haberman.columns)[:-1]):
    print(column)
    counts, edges = np.histogram(haberman[column], bins=10, density=True)
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(edges[1:], pdf, edges[1:], cdf)
    plt.xlabel(column)
    plt.show()




import numpy as np
survivalstatus_one = haberman.loc[haberman["SurvivalStatus"] == 'yes']
survivalstatus_two = haberman.loc[haberman["SurvivalStatus"] == 'no']

#Mean, Variance, Std-deviation,  
print("Means:")
for column in list(haberman.columns)[:-1]:
    print("Mean of {} for Survival Status == yes is {} ". format(column, np.mean(survivalstatus_one[column])))
    print("Mean of {} for Survival Status == no is {} ". format(column, np.mean(survivalstatus_two[column])))
    print('*'*50)

print("Medians:")

for column in list(haberman.columns)[:-1]:
    print("Median of {} for Survival Status == yes is {} ". format(column, np.median(survivalstatus_one[column])))
    print("Median of {} for Survival Status == no is {} ". format(column, np.median(survivalstatus_two[column])))
    print('*'*50)

print("Std Deviations: ")
for column in list(haberman.columns)[:-1]:
    print("Std. Deviation of {} for Survival Status == yes is {} ". format(column, np.std(survivalstatus_one[column])))
    print("Std. Deviation of {} for Survival Status == no is {} ". format(column, np.std(survivalstatus_two[column])))
    print('*'*50)




figure, axes = plt.subplots(1, 3, figsize=(20, 5))
for index, column in enumerate(list(haberman.columns)[:-1]):
    sns.boxplot( x='SurvivalStatus', y=column, data=haberman, ax=axes[index])
plt.show()  




from statsmodels import robust
print("\nQuantiles for Status survival type 1")
for column in list(haberman.columns)[:-1]:
    print("Quantiles of {} are {}".format(column, np.percentile(survivalstatus_one[column],np.arange(0, 100, 25))))
    print("90th Quantile of {} is {}".format(column, np.percentile(survivalstatus_one[column],90)))
    print("MAD of {} is {}".format(column,robust.mad(survivalstatus_one[column])))
    print('*'*50)

print("\nQuantiles for Status survival type 2")
for column in list(haberman.columns)[:-1]:
    print("Quantiles of {} are {}".format(column, np.percentile(survivalstatus_two[column],np.arange(0, 100, 25))))
    print("90th Quantile of {} is {}".format(column, np.percentile(survivalstatus_two[column],90)))
    print("MAD of {} is {}".format(column,robust.mad(survivalstatus_two[column])))
    print('*'*50)




figure, axes = plt.subplots(1, 3, figsize=(20, 5))
for index, column in enumerate(list(haberman.columns)[:-1]):
    sns.violinplot( x='SurvivalStatus', y=column, data=haberman, ax=axes[index])
plt.show() 




sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="SurvivalStatus", size=4)    .map(plt.scatter, "PatientAge", "AxillaryNodes")    .add_legend();
plt.show();




sns.pairplot(haberman, hue='SurvivalStatus', size=4)
plt.show()

