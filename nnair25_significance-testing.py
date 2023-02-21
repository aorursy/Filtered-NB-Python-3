#!/usr/bin/env python
# coding: utf-8



#importing header files
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors




#Checking a sample file
F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data1.csv")
#Dropping an unnecessary column
F=F.drop('Unnamed: 0',axis=1)
#Checking the sample data
print(F.head(5))
print(F.dtypes)




n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S1 obj' and F.loc[5,'subject identifier']=='a':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X1=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)

s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X1)




n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S1 obj' and F.loc[5,'subject identifier']=='c':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X2=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X2)




n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S2 match' and F.loc[5,'subject identifier']=='a':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X3=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X3)




n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S2 match' and F.loc[5,'subject identifier']=='c':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X4=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X4)




n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S2 nomatch,' and F.loc[5,'subject identifier']=='a':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X5=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X5)




n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'matching condition']=='S2 nomatch,' and F.loc[5,'subject identifier']=='c':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop")

X6=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)
s,ax=plt.subplots(figsize=(40,20))
sns.barplot(x='channel',y='sensor value',data=X6)




S1=pd.DataFrame({'Channel':X1['channel'], 'Alcohol':X1['sensor value'],'Control':X2['sensor value']})
S2Match=pd.DataFrame({'Channel':X3['channel'], 'Alcohol':X3['sensor value'],'Control':X4['sensor value']})
S2NoMatch=pd.DataFrame({'Channel':X5['channel'], 'Alcohol':X5['sensor value'],'Control':X6['sensor value']})




def brain_test(df, col1, col2, alpha):
    from scipy import stats
    import scipy.stats as ss
    import pandas as pd
    import statsmodels.stats.weightstats as ws
    
    n, _, diff, var, _, _ = stats.describe(df[col1] - df[col2])
    degfree = n - 1

    temp1 = df[col1].as_matrix()
    temp2 = df[col2].as_matrix()
    res = ss.ttest_rel(temp1, temp2)
      
    means = ws.CompareMeans(ws.DescrStatsW(temp1), ws.DescrStatsW(temp2))
    confint = means.tconfint_diff(alpha=alpha, alternative='two-sided', usevar='unequal') 
    degfree = means.dof_satt()

    index = ['DegFreedom', 'Difference', 'Statistic', 'PValue', 'Low95CI', 'High95CI']
    return pd.Series([degfree, diff, res[0], res[1], confint[0], confint[1]], index = index)   
    
def hist_brain_conf(df, col1, col2, num_bins = 30, alpha =0.05):
    import matplotlib.pyplot as plt
    ## Setup for ploting two charts one over the other
    fig, ax = plt.subplots(2, 1, figsize = (12,8))
    
    mins = min([df[col1].min(), df[col2].min()])
    maxs = max([df[col1].max(), df[col2].max()])
    
    mean1 = df[col1].mean()
    mean2 = df[col2].mean()
    
    tStat = brain_test(df, col1, col2, alpha)
    pv1 = mean2 + tStat[4]    
    pv2 = mean2 + tStat[5]
    
    ## Plot the histogram   
    temp = df[col1].as_matrix()
    ax[1].hist(temp, bins = 30, alpha = 0.7)
    ax[1].set_xlim([mins, maxs])
    ax[1].axvline(x=mean1, color = 'red', linewidth = 4)    
    ax[1].axvline(x=pv1, color = 'red', linestyle='--', linewidth = 4)
    ax[1].axvline(x=pv2, color = 'red', linestyle='--', linewidth = 4)
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel(col1)
    
    ## Plot the histogram   
    temp = df[col2].as_matrix()
    ax[0].hist(temp, bins = 30, alpha = 0.7)
    ax[0].set_xlim([mins, maxs])
    ax[0].axvline(x=mean2, color = 'red', linewidth = 4)
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel(col2)
    
    return tStat




hist_brain_conf(S1, 'Control','Alcohol')




hist_brain_conf(S2Match, 'Alcohol','Control')




hist_brain_conf(S2NoMatch, 'Control','Alcohol')




n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'subject identifier']=='a':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop1")

Alc=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)

n=0
for i in range(1,469):
    F=pd.read_csv("../input/SMNI_CMI_TRAIN/Data"+str(i)+".csv")
    #Converting objects to aprropriate DataTypes
    F['matching condition']= F['matching condition'].astype(str)
    F['subject identifier']= F['subject identifier'].astype(str)
    if F.loc[5,'subject identifier']=='c':
        n=n+1
        if n==1:
            Z= F
        else: 
            Z = Z.append(F)
print("End of Loop2")
Cont=Z[['channel','sensor value']].groupby('channel',as_index=False).mean().sort_values(by='channel',ascending=False)




Hypo=pd.DataFrame({'Channel':Alc['channel'], 'Alcohol':Alc['sensor value'],'Control':Cont['sensor value']})
hist_brain_conf(Hypo, 'Control','Alcohol')

