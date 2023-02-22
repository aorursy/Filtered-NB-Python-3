#!/usr/bin/env python
# coding: utf-8



import sys
print(sys.version)




import datetime
print(datetime.datetime.now())




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
savefig = {'bbox_inches': 'tight', 'dpi': 300}
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')




# Put the victim codes into a dictionary
with open('../input/Victim Codes.csv') as f:
    data = [line.split(',') for line in f.read().splitlines()]
    data = [d.strip() for row in data for d in row if d]
    data = [(float(d.split()[0]), ' '.join(d.split()[1:])) for d in data]
    victim_codes = dict(data)
    
# Load the main csv data
df = pd.read_csv('../input/Serial Killers Data.csv', encoding = 'ISO-8859-1')




# Map the victim types to the codes
df['Victim'] = df.VictimCode.map(victim_codes)




# Convert to datetimes
# Note: timestamps before 1677 are not supported by pandas and are set as NaN below

# These ones are easy
cols = ['birthyear', 'YearFirst', 'YearFinal']
for col in cols:
    df[col] = pd.to_datetime(df[col])

# These ones are more messy
cols = ['BirthDate', 'DateFirst', 'DateFinal', 'DateDeath']

def custom_datetime_filter(col):
    
    vals = []
    for val in df[col]:
        new_val = float('nan') # Datetimes are "guilty until proven innocent"
        if val != float('nan'):
            for fmt in ('%m/%d/%Y', '%Y'):
                try:
                    # If this works then we'll keep the datetime as is
                    pd.to_datetime(val, format=fmt)
                    new_val = val
                except:
                    continue
        vals.append(new_val)
    
    return vals

for col in cols:
    df[col] = pd.to_datetime(custom_datetime_filter(col))




# How many columns do we have?
len(df.columns)




df.head(1)




# The colors we'll use
colors = sns.color_palette("deep", 10)




# Parese Type column to determine motives
df['Motive'] = df.Type.apply(lambda x: str(x).split('-')[0])
df.Motive.value_counts()[:10]




def plot_hist(var, split_col, splits, weight_kills=False, match_type='eq'):
    ''' Plot a segmented histogram. '''
    
    fig, axes = plt.subplots(len(splits), figsize=(8, 8))
    for ax, s, c in zip(axes, splits, colors):
        
        if match_type == 'eq':
            mask = df[split_col] == s
        elif match_type == 'in':
            mask = df[split_col].apply(lambda x: s in x)
        else:
            print('Invalid match type, must be "eq" or "in".')
            return None

        if weight_kills:
            weights = df[mask].dropna(subset=[var]).NumVics.values
        else:
            weights = [1]*len(df[mask][var].dropna())
            
        df[mask][var].hist(bins=100, weights=weights,
                           alpha=0.65, label=s, ax=ax, color=c)
        
        ax.set_xlim(x1, x2)
        ax.legend(loc='upper left');
    plt.subplots_adjust(hspace=0.05)
    plt.suptitle('%s distribution split by %s' % (var, split_col) + ' (weighted by kills)'*weight_kills, y=0.93, fontsize=13)




x1, x2 = datetime.datetime.strptime('1900', '%Y'), datetime.datetime.now()    

motives = ['Enjoyment', 'FinancialGain', 'Anger', 'Criminal', 'GangActivity']
plot_hist('BirthDate', 'Motive', motives)
plot_hist('BirthDate', 'Motive', motives, weight_kills=True)




# Parese Type column to determine more detailed motives
def parse_Type_detail(x):
    try:
        x_ = str(x).split('-')[1]
    except:
        x_ = float('nan')
    return x_
df['MotiveDetail'] = df.Type.apply(parse_Type_detail)
df.MotiveDetail.value_counts()[:10]




motives = ['Rape', 'Robbery', 'Norape', 'Power', 'Revenge']
plot_hist('BirthDate', 'MotiveDetail', motives)
plot_hist('BirthDate', 'MotiveDetail', motives, weight_kills=True)




df.MethodDescription = df.MethodDescription.apply(str)
df.MethodDescription.value_counts()[:10]




methods = ['Shoot', 'Strangle', 'Stab', 'Bludgeon', 'Poison']
plot_hist('BirthDate', 'MethodDescription', methods, match_type='in')

gun_codes = {0: '', 1: 'Handgun', 2: 'Rifle', 3: 'Shotgun'}
def custom_map(x):
    try:
        x_ = ', '.join([gun_codes[int(code)] for code in str(x).split(',')])
    except:
        x_ = ''
    return x_
df['Gun'] = df.Gun.apply(custom_map)




df.Gun.value_counts()




methods = ['Handgun', 'Shotgun', 'Rifle']
plot_hist('BirthDate', 'Gun', methods, weight_kills=True, match_type='in')




df_ = df.groupby(['Motive', 'MethodDescription']).NumVics.sum().reset_index()

# Split up dataframe by method - conserving total NumVics
print('%d total victims' % df_.NumVics.sum())
new_rows = []
for row in df_.values:
    data = row[1].split(',')
    for d in data:
        new_rows.append([row[0], d, row[2]/len(data)])
df_ = pd.DataFrame(new_rows, columns=df_.columns)
print('%d total victims' % df_.NumVics.sum())

# Re-group and sort
df_ = df_.groupby(['Motive', 'MethodDescription']).NumVics.sum().reset_index().sort_values(['Motive', 'NumVics'], ascending=False)
df_.NumVics = df_.NumVics.apply(int)

# Remove NaNs
df_ = df_[df_.apply(lambda x: not any([x_=='nan' or x_=='' for x_ in x]), axis=1)]

# Print groups
for motive in df_.Motive.unique()[::-1]:
    df_print = df_[df_.Motive == motive].sort_values('NumVics', ascending=False).head(5)
    # Only print the noteworty ones
    if len(df_print) == 5:
        display(df_print)




# The whole ugly mess:    
df_.pivot_table(index='Motive', columns='MethodDescription', values='NumVics').fillna('')




# Assigning each person a label based on the number of methods used
# new column is 1 if greater than 2 methods else 0 if only 1 method
def assign_label(x):
    x_ = x.split(',')
    if len(x_) > 1:
        return 1
    return 0

df['MultipleMethods'] = df.MethodDescription.apply(assign_label)




print('%d%% of people used > 1 method' % (len(df[df['MultipleMethods'] == 1]['MultipleMethods']) / len(df) * 100))




col = 'YearsBetweenFirstandLast'
sns.distplot(df[df.MultipleMethods == 0].YearsBetweenFirstandLast.dropna(), bins=50, kde=True, norm_hist=True, label='1 method', color=colors[0])
sns.distplot(df[df.MultipleMethods == 1].YearsBetweenFirstandLast.dropna(), bins=50, kde=True, norm_hist=True, label='> 1 method', color=colors[1])
plt.legend()
plt.xlim(0, 30);




var = 'Age1stKill'
sns.distplot(df[df.MultipleMethods == 0][var].dropna(), bins=50, kde=True, norm_hist=True, label='1 method', color=colors[0])
sns.distplot(df[df.MultipleMethods == 1][var].dropna(), bins=50, kde=True, norm_hist=True, label='> 1 method', color=colors[1])
plt.legend()
plt.xlim(10, 60);




cols = ['US', 'England', 'S.Africa', 'Japan', 'Italy', 'Germany', 'India', 'Russia' 'Australia', 'Canada']
df_ = pd.DataFrame(df.groupby(['country', 'MultipleMethods']).Name.agg(lambda x: len(x)).rename('Counts')[cols])
# THIS IS UGLY - don't even bother looking at it
df_['Percent'] = list(map(lambda x: '%d%%' % x, np.array([[v1/(v1+v2)*100, v2/(v1+v2)*100] for v1, v2 in [df_.reset_index()[df_.reset_index()['country']==c].Counts.values for c in df_.reset_index()['country'].unique()]]).flatten()))
df_




motives = ('Anger',)
df_ = df[df.Motive.apply(lambda x: x in motives)].groupby(['Motive', 'Victim']).Name.agg(lambda x: len(x))
motives = ('Revenge',) 
df_ = pd.concat((df_, df[df.MotiveDetail.apply(lambda x: x in motives)].rename(columns={'Motive': '', 'MotiveDetail': 'Motive'}).groupby(['Motive', 'Victim']).Name.agg(lambda x: len(x))))
df_ = pd.DataFrame(df_.rename('Counts'))
df_.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values('Counts', ascending=False).head(3))




motives = ('FinancialGain',)
df_ = df[df.Motive.apply(lambda x: x in motives)].groupby(['Motive', 'Victim']).Name.agg(lambda x: len(x))
df_ = pd.DataFrame(df_.rename('Counts'))
df_.groupby(level=0, group_keys=False).apply(lambda x: x.sort_values('Counts', ascending=False).head(5))

