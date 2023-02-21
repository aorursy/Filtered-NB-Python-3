#!/usr/bin/env python
# coding: utf-8



# must for data analysis
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from matplotlib.pyplot import *

# useful for data wrangling
import io, os, re, subprocess

# for sanity
from pprint import pprint




def ca_law_enforcement_by_campus(data_directory):
    filename = 'ca_law_enforcement_by_campus.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        lines = f.readlines()
    
    header = ' '.join(lines[:6])
    header = re.sub('\n','',header)
    data = lines[6:]
    
    # Process each string in the list
    newlines = []
    for p in data:
        if( len(re.findall(',,,,',p))==0):
            newlines.append(p)

    # Combine into one long string, and do more processing
    one_string = '\n'.join(newlines)
    sio = io.StringIO(one_string)

    columnstr = header

    # Get rid of \r stuff
    columnstr = re.sub('\r',' ',columnstr)
    columnstr = re.sub('\s+',' ',columnstr)
    columns = columnstr.split(",")
    columns = [s.strip() for s in columns]

    df = pd.read_csv(sio,quotechar='"', names=columns, thousands=',')

    return df

def ca_offenses_by_campus(data_directory):
    filename = 'ca_offenses_by_campus.csv'

    # Load file into list of strings
    with open(data_directory + '/' + filename) as f:
        lines = f.readlines()
    
    # Process each string in the list
    newlines = []
    for p in lines[1:]:
        if( len(re.findall(',,,,',p))==0):
            # This is a weird/senseless/badly formatted line
            if( len(re.findall('Medical Center, Sacramento5',p))==0):
                newlines.append(p)

    one_line = '\n'.join(newlines)
    sio = io.StringIO(one_line)
    
    # Process column names
    columnstr = lines[0].strip()
    columnstr = re.sub('\s+',' ',columnstr)
    columnstr = re.sub('"','',columnstr)
    columns = columnstr.split(",")
    columns = [s.strip() for s in columns]
    
    # Load the whole thing into Pandas
    df = pd.read_csv(sio, quotechar='"', thousands=',', names=columns)
    
    return df




df_enforcement = ca_law_enforcement_by_campus('../input/')
df_offenses = ca_offenses_by_campus('../input/')




print(len(df_enforcement))
print(len(df_offenses))




for r in df_offenses['Campus']:
    if(type(r)==type(' ')):
        df_offenses['Campus'][df_offenses['Campus']==r].map(lambda x : re.sub(r'[0-9]$','',x))




df_campus = pd.merge(df_offenses, df_enforcement, 
                     on=[df_enforcement.columns[0],df_enforcement.columns[1],df_enforcement.columns[2]])




df_campus['Per Capita Law Enforcement Personnel'] = (df_campus['Total law enforcement employees'])/(df_campus['Student enrollment'])
df_campus['Law Enforcement Civilians Per Officer'] = (df_campus['Total civilians'])/(df_campus['Total officers'])

df_campus['Aggregate Crime'] = df_campus['Violent crime'] + df_campus['Property crime'] + df_campus['Arson']
df_campus['Per Capita Violent Crime'] = (df_campus['Violent crime'])/(df_campus['Student enrollment'])
df_campus['Per Capita Property Crime'] = (df_campus['Property crime'])/(df_campus['Student enrollment'])
df_campus['Per Capita Aggregate Crime'] = (df_campus['Violent crime'] + df_campus['Property crime'] + df_campus['Arson'])/(df_campus['Student enrollment'])

df_campus['Aggregate Crime Per Officer'] = (df_campus['Aggregate Crime'])/(df_campus['Total officers'])
df_campus['Violent Crime Per Officer'] = (df_campus['Violent crime'])/(df_campus['Total officers'])
df_campus['Property Crime Per Officer'] = (df_campus['Property crime'])/(df_campus['Total officers'])




# Start with the data we are going to cut up
data = df_campus['Student enrollment']

bins = [0, 0.20, 0.5, 0.80, 1.0]

# Here's what qcut looks like:
pd.qcut(data,bins).head()




group_names = ['Super Tiny','Small-ish','Large-ish','Massive']
df_campus['School size'] = pd.qcut(data,bins,labels=group_names)




pd.value_counts(df_campus['School size']).sort_index()




import seaborn as sns




sns.stripplot(x="School size", y="Aggregate Crime", data=df_campus, jitter=True)
title('Aggregate Crime vs Campus Size')
show()




sns.stripplot(x="School size", y="Per Capita Aggregate Crime", data=df_campus, jitter=True)
title('Per Capita Aggregate Crime vs Campus Size')
show()




tiny_sorted = (df_campus[df_campus['School size']=='Super Tiny'].sort_values('Aggregate Crime',ascending=False))[['University/College','Campus']]
print(tiny_sorted.iloc[0])




smallish_sorted = (df_campus[df_campus['School size']=='Small-ish'].sort_values('Aggregate Crime',ascending=False))[['University/College','Campus']]
print(smallish_sorted.iloc[0:4])




largeish_sorted = (df_campus[df_campus['School size']=='Large-ish'].sort_values('Aggregate Crime',ascending=False))[['University/College','Campus']]
print(largeish_sorted.iloc[0:4])




massive_sorted = (df_campus[df_campus['School size']=='Massive'].sort_values('Aggregate Crime',ascending=False))[['University/College','Campus']]
print(massive_sorted.iloc[0:2])




print(tiny_sorted.iloc[-1])




print(smallish_sorted.iloc[-3:-1])




print(largeish_sorted.iloc[-4:-1])




print(massive_sorted.iloc[-4:-1])




## Factor Plot: School Size and...?

Examining the trend of aggregate crime versus school size revealed grouping in the data. We can use a factor plot to explore other factors.




df_campus.columns




sns.stripplot(x="School size", y="Aggregate Crime",
               data=df_campus, jitter=True)
show()




unicol = df_campus['University/College']

university_categories = []
for (i,j) in (pd.value_counts(unicol)>1).iteritems():
    if j:
        # Compile a list of all College/University names with more than 1 campus
        university_categories.append(i)

## To filter out 1-campus schools, use this:
#df_multi_campus = df_campus[df_campus['University/College'].map(lambda x : x in university_categories)]

# To add 1-campus schools to an "Other" category, use this:
df_campus['UCtemp'] = df_campus['University/College'].map(lambda x : x if x in university_categories else "Other")




sns.lmplot(x="Student enrollment", y="Aggregate Crime",
               data=df_campus, hue="UCtemp")
show()




sns.lmplot(x="Student enrollment", y="Violent crime",
               data=df_campus, hue="UCtemp")
show()




df_campus.sort_values('Violent crime',ascending=False).iloc[0:2]




f, axes = subplots(1,3, figsize=(10, 3))

variables = ['Aggregate Crime','Violent Crime','Property Crime']

for ax,varlabel in zip(axes,variables):
    sns.stripplot(x="School size", y="Per Capita "+varlabel, data=df_campus, jitter=True, ax=ax)
    ax.set_title(varlabel+' vs Campus Size')
show()




label1 = ['University/College','Campus','Student enrollment']
label2 = ['Per Capita Aggregate Crime','Per Capita Violent Crime','Per Capita Property Crime']
tiny_schools = df_campus[df_campus['School size']=='Super Tiny']

for sort_label in label2:
    print("="*60)
    print("Schools Ranked By "+sort_label+":")
    pprint( tiny_schools.sort_values(sort_label, ascending=False).iloc[0:3][label1+label2].T )




plot(df_campus['Per Capita Law Enforcement Personnel'], df_campus['Per Capita Aggregate Crime'],'o')
xlabel('Per Capita Law Enforcement Personnel')
ylabel('Per Capita Aggregate Crime')
show()




print(df_campus[['University/College','Campus']][df_campus['Per Capita Law Enforcement Personnel']>0.005])




df_campus_filtered = df_campus[df_campus['Per Capita Law Enforcement Personnel']<0.005]
plot(df_campus_filtered['Per Capita Law Enforcement Personnel'], df_campus_filtered['Per Capita Aggregate Crime'],'o')
xlabel('Per Capita Law Enforcement Personnel')
ylabel('Per Capita Aggregate Crime')
show()




ratio_bins = [0.0,0.33,0.66,1.0]
ratio_data = df_campus['Law Enforcement Civilians Per Officer']
ratio_labels = ["More Civilians","Mixed","More Officers"]

df_campus['Law Enforcement Civilian Officer Ratio'] = 0.0
df_campus.loc[:,['Law Enforcement Civilian Officer Ratio']] = pd.qcut(ratio_data, ratio_bins, ratio_labels)




df_campus_filtered = df_campus[df_campus['Per Capita Law Enforcement Personnel']<0.005]
sns.lmplot(x="Per Capita Law Enforcement Personnel", y="Per Capita Aggregate Crime", 
           hue="Law Enforcement Civilian Officer Ratio",
           data=df_campus_filtered)
xlim([0.0,0.005])
ylim([0.00,0.05])
show()




sns.lmplot(x="Total law enforcement employees", y="Aggregate Crime", 
           hue="Law Enforcement Civilian Officer Ratio",
           data=df_campus)
show()

