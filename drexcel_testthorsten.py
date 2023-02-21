#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sys
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# prepare ExcelWriter for output
excel = pd.ExcelWriter('results.xlsx')




# ============================================
#          Choose the right columns
# ============================================
# the data set is huge, so we select only the columns of interest to save space

# standard columns first
pupils_cols = ['ParticipantName','MediaName','RecordingTimestamp','PupilLeft','PupilRight','ValidityLeft','ValidityRight']

# now all columns that contain the string 'AOI'
daten = pd.read_table("../input/Framing_IND_Inexp.tsv",nrows=1)
AOIs = []
for c in daten.columns:
    s = str(c)
    if (s.find('AOI')==0):
        AOIs.append(s)
spalten = pupils_cols.copy()
spalten.extend(AOIs.copy())
data = pd.read_table("../input/Framing_IND_Inexp.tsv", usecols=spalten)


# ============================================
#    Compress data for better performance
# ============================================

# -----------
#   AOIs
# -----------
# make a dictionary of AOIs and save it as csv
AOI_dict = dict(zip(AOIs,np.arange(len(AOIs))))
tabelle = {'ID': np.arange(len(AOIs)), 'AOI': AOIs}
AOI_table = pd.DataFrame(tabelle)
AOI_table.to_csv('AOI_table.csv', sep=';', decimal=',', header=True, index=False, columns=['ID','AOI'])
AOI_table.to_excel(excel, 'AOI_IDs', columns=['ID','AOI'], index=False)

# make an AOI column for later sorting (and to compress data to save RAM)
data['AOI_No'] = None
pupils_cols += ['AOI_No']
print(pupils_cols)
for aoi in AOIs:
    data.loc[data[aoi] == 1, 'AOI_No'] = AOI_dict[aoi]    
    
# ------------------------
#  Stimuli (MediaNames)
# ------------------------
# same data compression for the stimuli

Media = list(data['MediaName'].value_counts().index)
Media_dict = dict(zip(Media,np.arange(len(Media))))
tabelle = {'ID': np.arange(len(Media)), 'Medium': Media}
Media_table = pd.DataFrame(tabelle)
Media_table.to_csv('Media_table.csv', sep=';', decimal=',', header=True, index=False, columns=['ID','Medium'])
# Media_table.to_excel(excel, columns=['ID','Medium'])

data['Media_No'] = None
for medium in Media:
    # print(medium)
    data.loc[data['MediaName'] == medium, 'Media_No'] = Media_dict[medium]    
    # print(data.loc[data['MediaName'] == medium, 'Media_No'])
    
# show some information
print(Media_dict)




# function for calculating length of fixation
def time_spent(arr):
    return arr.max() - arr.min()




# choose only relevant subset of data
stimuli_cols = ['ParticipantName','RecordingTimestamp','MediaName']
stimuli = data[stimuli_cols]

# get rid of missing data rows
cleaned = stimuli.dropna() 

# Time in Stimulus
stimulus_times = cleaned.groupby(['MediaName','ParticipantName']).agg(time_spent)
# print(stimulus_times)

# output results to files
# print('Output to csv and Excel...')
#stimulus_times.to_csv('media_times.csv', sep=';', decimal=',', header=True)
#stimulus_times.to_excel(excel,'TimeInMedium')

# ---------
# plot
# ---------
# flatten the structure to return a normal table, otherwise plottin is impossible
stacked = stimulus_times.stack().reset_index()  
# the resulting table has a column with only 'RecordingTimestamp' in it
# this is useless, so drop it
stacked = stacked.drop(columns=['level_2']) 
# the column with time in stimulus is named 0, let's put a proper heading instead                                            
stacked.rename(columns={0:'TimeInStimulus'}, inplace=True)
# now plot with seaborn
fig = plt.figure()
fig.set_size_inches(15,6)
sns.swarmplot(x="MediaName", y="TimeInStimulus", data=stacked);




# choose only relevant subset of data
stimuli_cols = ['ParticipantName','RecordingTimestamp','AOI_No','MediaName']
stimuli = data[stimuli_cols]

# get rid of missing data rows
cleaned = stimuli.dropna() 

# Time in Stimulus
stimulus_times = cleaned.groupby(['MediaName','AOI_No','ParticipantName']).agg(time_spent)
print(stimulus_times)

print('Output is saved to csv and Excel...')
stimulus_times.to_csv('aoi_times.csv', sep=';', decimal=',', header=True)
stimulus_times.to_excel(excel, 'TimeInAOI', merge_cells=False)




# Time to first Fixation
stimulus_TTFF = cleaned.groupby(['ParticipantName','MediaName','AOI_No']).agg(min)
TTFF_sortiert = stimulus_TTFF.reset_index().sort_values(['ParticipantName','MediaName','RecordingTimestamp'],ascending=True).set_index(['ParticipantName','MediaName','AOI_No'])
print(TTFF_sortiert)

print('Output is saved to csv and Excel...')
TTFF_sortiert.to_csv('TTFF.csv', sep=';', decimal=',', header=True)
#TTFF_sortiert.to_excel(excel, 'TTFF', merge_cells=False)




print('=================================================')
print(' Calculating the Mean of both pupil data columns')
print('=================================================')

print('... checking data quality')
# check data quality of validity columns
print(data.ValidityLeft.isnull().value_counts())
print(data.ValidityRight.isnull().value_counts())

# Pupil data only: How is quality?
pupils = pd.DataFrame(data,columns=pupils_cols)
print(pupils.PupilLeft.isnull().value_counts())
print(pupils.PupilRight.isnull().value_counts())
pupils.head()

# cleaning
cleaned = pupils.dropna() # get rid of missing data rows
valide = cleaned[(cleaned['PupilLeft'] != '-1,00') & (cleaned['PupilRight'] != '-1,00')] # get rid of -1 (invalid data)
for col in ['PupilLeft','PupilRight']:  
    valide[col] = valide[col].str.replace(',','.') # replace comma by colon
    valide[col] = valide[col].astype(float)  # change data type to number for calculation
valide = valide[(valide['ValidityLeft']==0) | (valide['ValidityLeft']==4)]
valide['PupilMean'] = valide[['PupilLeft', 'PupilRight']].mean(axis=1)
print()
print('Cleaned Data with Mean')
print('-----------------------')
valide.info()
valide.head()

print()
print('------------------------------------------------------')
print("Avg. Pupil by AOI")
print('------------------------------------------------------')
print()
get_ipython().run_line_magic('matplotlib', 'inline')
#for aoi in AOIs:
#    thisAOIdata = valide[data['AOI_No'] == AOI_dict[aoi]]
#    grouped = thisAOIdata['PupilMean'].groupby(thisAOIdata['ParticipantName'])
#    means = grouped.mean()
#    if means.count() != 0:
#        print(aoi)
#        print('-------------------------')
#        print(means)
#        print('-------------------------')
#        print('Average for AOI: {0:8.2f}'.format(thisAOIdata['PupilMean'].mean()))
#        print('-------------------------')
#    print()
# plt.plot(thisAOIdata['PupilMean'])
# grouped = valide['PupilMean'].groupby(valide['ParticipantName'])
# means = grouped.mean()
# print(means)

print('Output is saved to csv and Excel...')
means = valide['PupilMean'].groupby([valide['AOI_No'],valide['ParticipantName']]).mean()
print(means)
means.to_csv('pupil_data.csv', sep=';', decimal=',', header=True)
means.to_excel(excel,'Pupils')
excel.save()

# cleaned = [(valide['ValidityLeft']==0) | (valide['ValidityLeft']==4)]
# valide.info()
# print(mitpunkt['PupilLeft'].value_counts())




data['MediaName'].value_counts()




valide.head(100).sort_values(by=['AOI_No','RecordingTimestamp'])




spalten = ['ParticipantName','MediaName','RecordingTimestamp','PupilLeft','PupilRight','ValidityLeft','ValidityRight']
typen = [np.object,np.object,np.int64,np.object,np.object,np.int8,np.int8]
types = dict(zip(spalten, typen))
daten = pd.read_table("../input/Framing_IND_Inexp.tsv",usecols=spalten,dtype=types,nrows=1000)
daten.info()




nums = data.select_dtypes(exclude=[object]) # take only numerical data
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # for plotting we need the library pyplot
# nums.hist(bins=50, figsize=(40,15)); # make histograms, figsize gives size of letters

# Doppelklick auf die Grafik macht die Zeichnungen groß!




# function for calculating length of fixation
def dauer(spalte):
    return np.max(spalte) - np.min(spalte)

# list (set) of participants
names = list(set(data.ParticipantName))

# list of AOIs
aois = []
for c in data.columns:
    s = str(c)
    if (s.find('AOI')==0):
        aois.append(s)
aois

# loop over all names:
for guy in names:
    print("================================================================
    print("Participant: ", guy)

    # loop over all media names (stimuli)
    erg = data.MediaName.value_counts()
    for medium in erg.index:
        print("---------------------------")
        print(medium)
        
        # time in stimulus
        auswahl = data[(data['MediaName']==medium) & (data['ParticipantName']==guy)]
        blickdauer = dauer(auswahl['RecordingTimestamp'])
        print("Time in Stimulus: ", blickdauer)
        print("---------------------------")
    
        # loop over all AOIs
        for AOI in aois:
            auswahl = data[(data['MediaName']==medium) & (data[AOI]==1) & (data['ParticipantName']==guy)]
            blickdauer = dauer(auswahl['RecordingTimestamp'])
            if blickdauer > -1:
                print(AOI,": ",blickdauer)




test = data[data.ParticipantName=='P11']
test.head()




daten = data # pd.read_csv("../input/Framing_IND_Inexp.tsv",'\t')




# let's have a look at the first few lines of data
daten.head()




daten['MediaName'].value_counts()





# get some info about data types and missing values
daten.info()




# now let's see if we get some nice graphics for the numerical data
nums = daten.select_dtypes(exclude=[object]) # take only numerical data




# let's get some basic information on our numerical data
nums.describe()





get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # for plotting we need the library pyplot
# nums.hist(bins=50, figsize=(40,15)); # make histograms, figsize gives size of letters




auswahl = daten[(daten['ParticipantName']=='P11') & (daten['RecordingTimestamp']==5)]
auswahl.describe()




print(daten['MediaName'].value_counts())
auswahl = daten[(daten['MediaName']=='Q2.png') & (daten['AOI[Question]Hit']==1)]
# Diff. von min/max Wert von RecordingTimestamp ist Blickdauer oder aber von erstem Auftreten das AOI[]Hit bis letztem
# auf jeden Fall pro Proband und pro MediaName
print("Dauer: ",np.max(auswahl['RecordingTimestamp']) - np.min(auswahl['RecordingTimestamp']))
auswahl.describe()

