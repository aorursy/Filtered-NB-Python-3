#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../input/LeagueofLegends.csv')




data.head()




print('Blue wins:', data.bResult.sum())
print('Red wins:', data.rResult.sum())




redWinsLength = data.groupby('gamelength').rResult.sum()
blueWinsLength = data.groupby('gamelength').bResult.sum()

plt.plot(redWinsLength.index, redWinsLength, 'r-')
plt.plot(blueWinsLength.index, blueWinsLength, 'b-')




redWins = data[data.rResult == 1]
blueWins = data[data.bResult == 1]

sns.distplot(redWins.gamelength, color='red', hist=False)
sns.distplot(blueWins.gamelength, color='blue', hist=False)




goldData = pd.read_csv('../input/goldValues.csv')
# Create minute column names for pd.melt()
minutes = ['min_' + str(x + 1) for x in range(81)]
goldData = pd.melt(goldData, id_vars=['MatchHistory', 'NameType'], value_vars=minutes, 
                   var_name='minute', value_name='gold')
# Changet the minute variable into a integer.
goldData.minute = goldData.minute.str.strip('min_').astype(int)
goldData.head()




blueGold = goldData[goldData.NameType == 'goldblue'].groupby('minute').gold.mean()
redGold = goldData[goldData.NameType == 'goldred'].groupby('minute').gold.mean()
goldDiff = goldData[goldData.NameType == 'golddiff'].groupby('minute').gold.mean()

plt.plot(blueGold, 'b-')
plt.plot(redGold, 'r-')
plt.plot(goldDiff, 'g-')
plt.xlabel('Minute')
plt.ylabel('Gold')




goldData.NameType.unique()




sections = ['Top', 'Jungle', 'Middle', 'ADC', 'Support']
num_sections = len(sections)

for i in range(num_sections):
    plt.figure(i)
    plt.plot(goldData[goldData.NameType == 'goldblue' + sections[i]].groupby('minute').gold.mean(), 'b-')
    plt.plot(goldData[goldData.NameType == 'goldred' + sections[i]].groupby('minute').gold.mean(), 'r-')
    plt.xlabel('Minute')
    plt.ylabel('Gold')
    plt.title(sections[i])




goldJungleDiff = goldData[goldData.NameType == 'goldblueJungle'].groupby('minute').mean() -                  goldData[goldData.NameType == 'goldredJungle'].groupby('minute').mean()
    
plt.plot(goldJungleDiff, 'g-')
plt.ylabel('Gold Diff')
plt.xlabel('Minute')
plt.title('Difference in gold from the Jungle')




killData = pd.read_csv('../input/deathValues.csv')
killData.head()




blueDeaths = killData[killData.TeamColor == 'Blue']
redDeaths = killData[killData.TeamColor == 'Red']
blueDeaths = blueDeaths[blueDeaths.Time.notnull()]
redDeaths = redDeaths[redDeaths.Time.notnull()]

sns.distplot(blueDeaths.Time, kde=False, color='blue')
sns.distplot(redDeaths.Time, kde=False, color='red')




objData = pd.read_csv('../input/objValues.csv')

objData.head()




nums = ['num_' + str(x) for x in range(1, 17)]
objData = pd.melt(objData, id_vars=['MatchHistory', 'ObjType'], value_vars=nums)




objData = pd.read_csv('../input/objValues.csv')
objData.head()




nums = ['num_' + str(x) for x in range(1, 17)]
objDataMelted = pd.melt(objData, id_vars=['ObjType'], value_vars=nums)
objDataMelted = objDataMelted[objDataMelted.value.notnull()]




sns.distplot(objDataMelted[objDataMelted.ObjType == 'bTowers'].value, kde=False, color='blue')
sns.distplot(objDataMelted[objDataMelted.ObjType == 'rTowers'].value, kde=False, color='red')

