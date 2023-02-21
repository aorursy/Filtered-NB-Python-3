#!/usr/bin/env python
# coding: utf-8



# Imports
import sys
import os
import urllib, base64
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# Define input files
datasetFolder = '../input//'
files = [
    '19.07.19_travels_Bonn.csv',
    '19.07.19_travels_Koblenz.csv',
    '19.07.19_travels_Koeln.csv',
]




def loadFile( _datasetFolder, _fileName ):
    fullPathFile = _datasetFolder + _fileName
    df = pd.read_csv( fullPathFile, engine='python', quoting=1, skip_blank_lines=True,
                     sep=",", quotechar='"',
                     encoding ='utf-8', dtype={'TSI': 'object'}
                    )
    return df.replace(np.nan, '', regex=True)




dfsDay = [
    loadFile( datasetFolder, files[0]),
    loadFile( datasetFolder, files[1]),
    loadFile( datasetFolder, files[2])
]




dfsDay[0].head(1)




# Select the index for the next steps
cityIndex = 2




# Extract City name and code
fullCityCode = dfsDay[ cityIndex ].iloc[0]['TSC']
cc = fullCityCode.split('%23')
cityCode = cc[1]
cityName = urllib.parse.unquote(cc[0])

cityCode, cityName




# Extract report date
reportDate = '20' + dfsDay[ cityIndex ].iloc[0]['TID']
reportDayWeekNum =  dt.datetime.strptime( reportDate, '%Y.%m.%d').weekday() # 0 Monday - 6 Sunday
reportDayWeekName =  dt.datetime.strptime( reportDate, '%Y.%m.%d').strftime("%A")

reportDate, reportDayWeekNum, reportDayWeekName




'''
In older versions the delay (TA) was a int. Now is a hh:mm.
It´s necessary to calculate the delay in minutes (TAc)
'''
def calculateDelay( row ):
    delay = 0
    if row['TA'] is not '':
        taHour = dt.datetime.strptime( str(row['TA']), '%H:%M')
        titHour = dt.datetime.strptime( str(row['TIT']), '%H:%M')
        delay = divmod( ( taHour - titHour ).total_seconds() , 60)[0]  # delay in minutes
    return int(delay)




def extractDataColumns( _df ):
    _df['TAc'] = _df.apply( calculateDelay, axis=1 )
    dfSort = _df.sort_values(['TIT', 'TIN'], ascending=[1, 1])
    dfF = pd.concat( [
            dfSort['TIN'],
            dfSort['TAc'],
            dfSort['TIT'].apply( lambda t: 
                                int( dt.datetime.strptime( t, '%H:%M' ).strftime('%H') )
            )
        ], axis=1, keys=['train', 'delay', 'hour'] ) \
        .reset_index()
    return dfF




dfBaseSorted = [
    extractDataColumns(dfsDay[0]),
    extractDataColumns(dfsDay[1]),
    extractDataColumns(dfsDay[2])
]

dfBaseSorted[0].head(2)




dfF = dfBaseSorted[ cityIndex ]




# Filter/exclude by train-type ?
dfF = dfF[ ~(dfF["train"].str.contains("S")) ]




g = dfF.groupby(['train','hour'])
#g.size()
#dfG = g.agg({'delay': ['min', lambda x: x.max(), 'max' ]})

dfFG = g.agg({'delay': ['max' ]})

dfFG.reset_index( inplace=True)
dfFG.columns = ['train','hour','delay_max']




dfFGS = dfFG.sort_values(['hour'], ascending=[1]).reset_index(drop=True)
dfFGS.head(5)




f = {'train' : ['count'],
       'delay_max': np.count_nonzero,
       'delay_max' : np.max,
   }
dfFGSM = dfFGS.groupby('hour').agg(f)
dfFGSM.columns = ['train_count', 'delay_max_minutes']
dfFGSM




dfFGSM180 = dfFGSM.transpose()
dfFGSM180




fig, ax = plt.subplots(figsize=(18,7))
plt.style.use('seaborn')
dfFGSM.plot(ax=ax, kind='bar')
plt.title('Trains and Minutes delay in ' + cityName + ' - ' + reportDate)
plt.show()




#plotName = 'delays-03-cityhours__' + cityName + '-' + str(reportDate)
#fig.savefig( 'YOUR/LOCAL/PATH//' + plotName + '.png', dpi=125)




reportCityDateTrains = {
    'cityCode'  : cityCode,
    'cityName'  : cityName,
    'reportDate': reportDate,
    'reportDayWeek': reportDayWeekNum,
    'df' : dfFGSM180
}
reportCityDateTrains

