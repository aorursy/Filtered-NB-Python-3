#!/usr/bin/env python
# coding: utf-8



#Load packages
import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans




#Read in and clean the data 
powerlift = pd.read_csv('../input/openpowerlifting.csv')

select_columns = [
       'MeetID', 'Name', 'Sex', 'Equipment',
       'BodyweightKg', 'WeightClassKg', 'BestSquatKg',
       'BestBenchKg',  'BestDeadliftKg',
       'TotalKg', 'Wilks']

powerlift = powerlift[select_columns]

powerlift[['BestSquatKg', 'BestBenchKg', 'BestDeadliftKg']] =             powerlift[['BestSquatKg', 'BestBenchKg', 'BestDeadliftKg']].fillna(0)

powerlift['TotalKg'] = powerlift['BestSquatKg'] +                         powerlift['BestBenchKg'] + powerlift['BestDeadliftKg']

#remove nan in BodyweightKg row less than 1% is affected by this
powerlift = powerlift.dropna(subset=['BodyweightKg'])

powerlift.info()




#functions for wilks score
def male_wilks(weight, totalkg):
    coeff = 500 / (-216.0475144 + 
                   (16.2606339 * weight) + 
                   (-0.002388645 * weight**2) + 
                   (-0.00113732 * weight**3) + 
                   (7.01863E-06 * weight**4) + 
                   (-1.291E-08 * weight**5))
    wilks = round(coeff * totalkg, 2)
    return wilks


def female_wilks(weight, totalkg):
    coeff = 500 / (594.31747775582 +
                  (-27.23842536447 * weight) +
                  (0.82112226871 * weight**2) +
                  (-0.00930733913 * weight**3) +
                  (4.731582E-05 * weight**4) +
                  (-9.054E-08 * weight**5))
    wilks = round(coeff * totalkg, 2)
    return wilks


#recalculate wilks column to fill in Nan
mask = (powerlift['Sex'] != 'F')
powerlift_valid = powerlift[mask]
powerlift.loc[mask, 'Wilks'] = male_wilks(powerlift_valid['BodyweightKg'], 
                                          powerlift_valid['TotalKg'])

mask = (powerlift['Sex'] != 'M')
powerlift_valid = powerlift[mask]
powerlift.loc[mask, 'Wilks'] = male_wilks(powerlift_valid['BodyweightKg'], 
                                          powerlift_valid['TotalKg'])

powerlift.info()





#remove duplicate rows
powerlift = powerlift.drop_duplicates().reset_index(drop=True)

#remove nan for WeightClassKg rows less than 1% is affected by this
powerlift = powerlift.dropna(subset=['WeightClassKg'])

#aggergate categorical variables WeightClass into federation
IPF = 'IPF'
Other_Feds = 'Not IPF'
not_common ='Abnormal Federation'

IPF_weightclass = ['47', '57', '59', '63', '66', '72', '74', '83', '84', '84+',
                   '93', '105', '120', '120+']
other_fed_weightclass = ['44', '48','56', '60', '67.5', '75', '82.5', '90', 
                         '90+', '100', '110', '125', '140', '140+']
abnormal_weightclass = ['30', '34', '35', '39', '40', '43', '50', '53', '60+', 
                        '67.5', '67.5+', '70', '70+', '75+', '80', '83+', '84+', 
                        '90+', '95', '155', '100+', '110+', '117.5', '125',
                        '125+', '145']

powerlift.loc[powerlift['WeightClassKg'].isin(IPF_weightclass),
             'Federation'] = IPF
powerlift.loc[powerlift['WeightClassKg'].isin(other_fed_weightclass),
             'Federation'] = Other_Feds
powerlift.loc[powerlift['WeightClassKg'].isin(abnormal_weightclass),
             'Federation'] = not_common


#determine which federation 52kg belongs because IPF and Not IPF have this weightclass
#creats a list of unique MeetID numbers for IPF and Not IPF

IP_Fed = powerlift[powerlift['Federation'] == 'IPF']
IP_Fed = IP_Fed['MeetID'].unique().tolist()

NotIP_Fed = powerlift[powerlift['Federation'] == 'Not IPF']
NotIP_Fed = NotIP_Fed['MeetID'].unique().tolist()
NotIP_Fed.extend([531, 1263, 2074])

powerlift.loc[powerlift['MeetID'].isin(IP_Fed),
             'Federation'] = IPF
powerlift.loc[powerlift['MeetID'].isin(NotIP_Fed),
             'Federation'] = Other_Feds

powerlift.info()




#seperate categories into quartiles

Squat_weight = ['Very Light Squat', 'Light Squat',
               'Moderate Squat', 'Heavy Squat', 
                'Very Heavy Squat']

powerlift['Squat Weight'] = pd.qcut(powerlift['BestSquatKg'], 5,
                                   Squat_weight)

Bench_weight = ['Very Light Bench', 'Light Bench',
               'Moderate Bench', 'Heavy Bench',
               'Very Heavy Bench']

powerlift['Bench Weight'] = pd.qcut(powerlift['BestBenchKg'], 5, 
                                   Bench_weight)

Deadlift_weight = ['Very Light DL', 'Light DL', 
                  'Moderate DL', 'Heavy DL',
                  'Very Heavy DL']

powerlift['Deadlift Weight'] = pd.qcut(powerlift['BestDeadliftKg'], 5,
                                      Deadlift_weight)

Total_weight = ['Very Light Total', 'Light Total', 
               'Moderate Total', 'Heavy Total', 
               'Very Heavy Total']

powerlift['Total Weight'] = pd.qcut(powerlift['TotalKg'], 5, 
                                   Total_weight)

WILKS = ['Very Low WILKS', 'Low WILKS', 
         'Moderate WILKS', 'Hight WILKS', 
         'Very Hight WILKS']

powerlift['WILKS Level'] = pd.qcut(powerlift['Wilks'], 5, 
                                  WILKS)
powerlift.info()




cluster_columns = ['BodyweightKg', 'BestSquatKg', 'BestBenchKg',
                  'BestDeadliftKg', 'TotalKg', 'Wilks']

scaler = preprocessing.MaxAbsScaler()


powerlift_cluster = scaler.fit_transform(powerlift[cluster_columns])
powerlift_cluster = pd.DataFrame(powerlift_cluster, 
                                columns=cluster_columns)

#cluster function


def kmeans_cluster(df, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=1)
    clusters = model.fit_predict(df)
    cluster_results = df.copy()
    cluster_results['Cluster'] = clusters
    return cluster_results

#summarizes clusters


def summarize_clustering(results):
    cluster_size = results.groupby(['Cluster']).size().reset_index()
    cluster_means = results.groupby(['Cluster'],
                                   as_index=False).mean()
    cluster_summary = pd.merge(cluster_size, 
                              cluster_means, on='Cluster')
    return cluster_summary


cluster_results = kmeans_cluster(powerlift_cluster, 5)
cluster_summary = summarize_clustering(cluster_results)

sns.heatmap(cluster_summary[cluster_columns].transpose(),
           annot=True)
plt.show()




pd.options.mode.chained_assignment = None  #to remove warnings

cluster_results['Cluster Name'] = ''
cluster_results['Cluster Name'][cluster_results['Cluster']==0] =                 'Average and Average'
cluster_results['Cluster Name'][cluster_results['Cluster']==1] =                 'Average and Strong'
cluster_results['Cluster Name'][cluster_results['Cluster']==2] =                 'Average and Weak'
cluster_results['Cluster Name'][cluster_results['Cluster']==3] =                 'Heavy and Strong'
cluster_results['Cluster Name'][cluster_results['Cluster']==4] =                 'Heavy and Weak'

powerlift = powerlift.reset_index().drop('index', axis=1)
powerlift['Cluster Name'] = cluster_results['Cluster Name']
powerlift.info()
powerlift.head()




#Function to help plot barplots for categorical columns
def agg_count(df, group_field):
    grouped = df.groupby(group_field, 
                        as_index=False).size()
    grouped = pd.DataFrame(grouped).reset_index()
    grouped.columns = [group_field, 'Count']
    return grouped




powerlift_females = powerlift[powerlift['Sex']== 'F']
female_count = agg_count(powerlift_females, 'WILKS Level')
sns.barplot(data=female_count, x='Count', y='WILKS Level')
plt.show()




powerlift_males = powerlift[powerlift['Sex']== 'M']
male_count = agg_count(powerlift_males, 'WILKS Level')
sns.barplot(data=male_count, x='Count', y='WILKS Level')
plt.show()




def pivot_count(df, rows, columns, calc_field):
    df_pivot = df.pivot_table(values=calc_field,
                             index=rows,
                             columns=columns,
                             aggfunc=np.size).dropna(axis=0, how='all')
    return df_pivot




powerlift['MeetID'].value_counts().sort_index().plot.line()




powerlift['Sex'].value_counts().plot.bar()




powerlift['Equipment'].value_counts().plot.bar()




powerlift['WeightClassKg'].value_counts().head(25).sort_index().plot.bar()




powerlift['WeightClassKg'].value_counts().tail(25).sort_index().plot.bar()




nom_categories = ['BodyweightKg', 'BestSquatKg', 'BestBenchKg', 'BestDeadliftKg', 'TotalKg', 'Wilks']

rows= 2
cols=3
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

counter = 0

for i in range(rows):
    for j in range(cols): 
        if counter < len(nom_categories):
            sns.distplot(powerlift[nom_categories[counter]], 
                         ax = axes[i][j], axlabel = nom_categories[counter].capitalize())
            counter += 1
            
plt.tight_layout()
plt.show()

