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




import seaborn as sns; sns.set(style="whitegrid", color_codes=True)
import matplotlib.pyplot as plt
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')

#load the dataset
hr0 = pd.read_csv('../input/HR_comma_sep.csv'); hr0.head()




dupes = hr0[hr0.duplicated(keep=False)]
print('There are {} duplicates out of total {} rows.'.format(dupes.shape[0], hr0.shape[0]))




#drop duplicates
hr0 = hr0.drop_duplicates()
hr0.shape
print('Without duplicates the dataset contains {} rows.'.format(hr0.shape[0]))

#copy original dataset
hr = hr0.copy(deep=True)




#function that creates a generic dictionary for a given tidy data
def create_dict(df,ncol_cat=20):
    '''This function takes a data frame follewed by 
    the threshold number of unique values in a single column
    to define categorical data type
    and returns a generic data frame dictionary'''
    
    df.columns=df.columns.str.lower()
    df_dict=pd.DataFrame(data=df.dtypes, columns=['type'])

    df_dict.index.name = 'column_name'
    df_dict['n_unique'] = df.nunique()

    type_info=[]
    range_info=[]
    for i in range(len(df_dict)):  
        range_info.append([min(df[df_dict.index[i]]),max(df[df_dict.index[i]])])
        if (df_dict.n_unique.values[i] > ncol_cat) & (df_dict.type.values[i] != 'object'):
            type_info.append('continuous')        
        else:
            type_info.append('categorical')
            #change data type to categorical to reduce memory usage
            df[df_dict.index[i]] = df[df_dict.index[i]].astype('category')
            
    df_dict['type_info'] = type_info
    df_dict['range_info'] = range_info
    df_dict['missing_values'] = df.isnull().sum()
    df_dict.sort_values(by = 'n_unique', inplace=True)

    reorder_cols=['type','type_info','n_unique','range_info','missing_values']
    df_dict = df_dict[reorder_cols]
    return df_dict




hr_dict = create_dict(hr)
display(hr_dict)

#assign an ordered category manually
hr.salary = pd.Categorical(hr.salary,categories=['low','medium','high'],ordered=True)




#function for a quick univariate analysis
def univar_plots(df,scols,sns_plottype,ncol_max=3):
    '''This function takes a data frame follewed by 
    the list of selected columns and plot type of sns library
    appropriate for univariate analysis and returns a figure 
    with the respective graphs for the selected data'''
    
    num_cols=len(scols)
    num_c = min([num_cols,ncol_max]); num_r = max([num_cols//ncol_max,1])
    
    if (num_r*num_c < num_cols)&(num_cols>ncol_max):
        num_r+=1

    fig, axes = plt.subplots(num_r, num_c, figsize=(min(16,num_c*4),ncol_max*num_r), sharex=False, sharey=False)

    i=0;j=0;k=0

    while (i < num_r):
        while (j < num_c) & (k < len(scols)):
            if num_r > 1:
                getattr(sns,sns_plottype)(y=scols[k], data=df, ax=axes[i,j])                   
            else:
                getattr(sns,sns_plottype)(y=scols[k], data=df, ax=axes[j])
                    
            k += 1; j += 1
        j = 0; i += 1

    fig.tight_layout()




#count plots for all categorical variables
univar_plots(hr,hr.columns[hr.dtypes == 'category'],sns_plottype='countplot',ncol_max=4)




#statistics for categorical variables
hr_desc = hr.describe(include=['category']).drop(['count','freq'])
scols_count = hr.columns[hr.dtypes == 'category']
#add percentege of most and least frequent value
hr_desc.loc['top_perc',:] = [hr[col].value_counts(normalize=True).max().round(2) for col in scols_count]
hr_desc.loc['bottom',:] = [hr[col].value_counts(normalize=True).idxmin() for col in scols_count]; 
hr_desc.loc['bottom_perc',:] = [hr[col].value_counts(normalize=True).min().round(2) for col in scols_count]

hr_desc




#to check the distributions for numerical variables use the violinplot
scols_num = hr.select_dtypes(exclude=['category']).columns
univar_plots(hr,scols_num,sns_plottype='violinplot',ncol_max=3)   




#search for outliers via box plots
univar_plots(hr,scols_num,sns_plottype='boxplot',ncol_max=3)   




#statistics summary for the continuous variables
hr.describe(include=['number']).drop(['count']).T




#fuction for quick bivariate analysis of all categorical vs categorical data
def bivar_heatmap_plots(df,scols,ncol_max=3):
    '''This function takes a data frame follewed by 
    the list of selected columns of categorical type
    and returns all combinations of heatmaps between 
    them listing the percentage values of occurances on top'''

    num_cols=len(scols); Nplots=int(num_cols*(num_cols-1)/2)
    num_c = min([Nplots,ncol_max]); num_r = max([Nplots//ncol_max,1])
    
    if (num_r*num_c < Nplots)&(Nplots>ncol_max):
        num_r+=1
    
    fig, axes = plt.subplots(num_r, num_c, figsize=(min(16,num_c*6),ncol_max*num_r), sharex=False, sharey=False)

    ik=0;jk=0
    for i in range(num_cols):      
        for j in range(i+1,num_cols):
            #calculate percentage (in %) of occurances
            col1_col2 = hr.groupby([scols[i],scols[j]]).size().unstack(scols[i]).div(hr.shape[0]).mul(100).round(1)
            cmap = sns.cubehelix_palette(10, start=jk, rot=0, dark=0.1, light=.9,as_cmap=True)
            
            if num_r > 1:
                sns.heatmap(col1_col2,annot=True,fmt=".1f",cmap=cmap,ax=axes[ik,jk])
            elif (num_r == 1)&(num_c == 1):
                sns.heatmap(col1_col2,annot=True,fmt=".1f",cmap=cmap);
            else:
                sns.heatmap(col1_col2,annot=True,fmt=".1f",cmap=cmap,ax=axes[jk])
                
            if jk < (num_c-1):
                jk += 1
            else:
                jk = 0; ik +=1
    fig.tight_layout()




#heatmap plots for all categorical variables
bivar_heatmap_plots(hr,hr.columns[hr.dtypes == 'category'],ncol_max=3)




col1='number_project';col2='time_spend_company';
scols_num = hr.select_dtypes(exclude=['category']).columns

fig, axes = plt.subplots(1, len(scols_num), figsize=(16,4), sharex=False, sharey=False)

for i in range(len(scols_num)):
    cc_heat = hr.groupby([col1,col2])[scols_num[i]].mean().unstack(col1)
    cmap = sns.cubehelix_palette(10, start=i, rot=0, dark=0.1, light=.9,as_cmap=True)
    sns.heatmap(cc_heat,annot=True,fmt=".1f",cmap=cmap,ax=axes[i])
    axes[i].set_title(scols_num[i])




scols_con = hr.select_dtypes(include=['number']).columns

sns.pairplot(hr,diag_kind='kde',vars=scols_con,hue='left',
                size=3,markers=["o", "s"],plot_kws={"s": 10},palette='cubehelix')




#find boundaries for satisfaction level
sort1 = pd.DataFrame(hr.groupby('left')['satisfaction_level'].value_counts().loc[1])
sort1.columns = ['count_num']
sort1.reset_index(inplace=True)
sort1.sort_values(by=['satisfaction_level'],inplace=True)

#find boundaries for last evaluation
sort2 = pd.DataFrame(hr.groupby('left')['last_evaluation'].value_counts().loc[1])
sort2.columns = ['count_num']
sort2.reset_index(inplace=True)
sort2.sort_values(by=['last_evaluation'],inplace=True)

#visualize 
fig, axes = plt.subplots(1, 2, figsize=(12,4), sharex=False, sharey=False)

sns.countplot(x='satisfaction_level', data=hr0[hr0.left==1],ax=axes[0])
sns.countplot(x='last_evaluation', data=hr0[hr0.left==1],ax=axes[1])

for i in range(2):
    axes[i].set_title('Distribution for employee who left')

    for ind, label in enumerate(axes[i].get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)




#there is an abrapt change for the following regions 
#show numbers:
#print('Find boundaries based on satisfaction level: \n',sort1[sort1.count_num > 10])
sat1=[0.09,0.11];sat2=[0.36,0.46];sat3=[0.72,0.92]

#show numbers:
#print('Find boundaries based on last evaluation: \n',sort2[sort2.count_num > 10])
last1=[0.77,1.0];last2=[0.45,0.57];last3=[0.8,1.0]




#set the conditions to indentify people who left
cond1 = (hr0.satisfaction_level >= 0.09) & (hr0.satisfaction_level <= 0.11) &        (hr0.last_evaluation >= 0.77)
group1 = hr0[cond1]
display(group1.left.value_counts(normalize=True))

cond2 = (hr0.satisfaction_level >= 0.36) & (hr0.satisfaction_level <= 0.46) &        (hr0.last_evaluation >= 0.45) & (hr0.last_evaluation <= 0.57)
group2 = hr0[cond2]
display(group2.left.value_counts(normalize=True))

cond3 = (hr0.satisfaction_level >= 0.72) & (hr0.satisfaction_level <= 0.92) &        (hr0.last_evaluation >= 0.8)
group3 = hr0[cond3]
display(group3.left.value_counts(normalize=True))




#refine 3rd condition
display(group3.groupby('number_project')['left'].value_counts())
display(group3.groupby('time_spend_company')['left'].value_counts())




cond3_mod = cond3 & ((hr0.time_spend_company == 5) | (hr0.time_spend_company == 6))
group3_mod = hr0[cond3_mod]
display(group3_mod.left.value_counts(normalize=True))




#combine all condition to build a criteria for employee to leave.
group_left= hr0[cond1|cond2|cond3_mod]
group_left.left.value_counts(normalize=True)






