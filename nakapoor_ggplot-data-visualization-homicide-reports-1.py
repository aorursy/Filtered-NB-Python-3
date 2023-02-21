#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ggplot import *

#import seaborn as sns #visualization
#sns.set_style('whitegrid')




hdata = pd.read_csv('../input/database.csv')




#Display features available in the Dataset
print(hdata.columns.values)




#Preview the Data (Beginning)
pd.set_option('display.max_columns', None)
hdata.head(10)




hdata.info()




#Convert Categorical Features that should be numerics for analysis
hdata['Perpetrator Age'] = pd.to_numeric(hdata['Perpetrator Age'], errors='coerce')




#Review the distribution of the numerical features
hdata.describe()




# Removing Record ID from the data set
data_1=hdata.drop(['Record ID','Agency Code','Agency Name','Agency Type','Incident','Victim Ethnicity','Perpetrator Ethnicity','Record Source'], 1)




#Review the distribution of the categorial features
data_1.describe(include=['O'])




print(data_1.columns.values)




print(len(np.unique(data_1['Crime Type'])))
print(len(np.unique(data_1['Victim Race'])))
print(len(np.unique(data_1['Perpetrator Race'])))
print(len(np.unique(data_1['Relationship'])))
print(len(np.unique(data_1['Weapon'])))




#print(len(np.unique(hdata['Agency Name'])))
print((np.unique(data_1['Crime Type'])))
print((np.unique(data_1['Victim Race'])))
print((np.unique(data_1['Perpetrator Race'])))
print((np.unique(data_1['Relationship'])))
print((np.unique(data_1['Weapon'])))




# ploting crime type
ggplot(data_1, aes('Crime Type')) + geom_bar()




# ploting crime type to show the victims sex statistics
ggplot(data_1, aes('Crime Type', fill = 'Victim Sex')) + geom_bar(position = "stack")





# From the above plot we can see that 'Murder or Manslaughter' occurs more often than 'Manslaughter by Negligence.'
# therefor Extracting the data based on Crime type  = 'Murder or Manslaughter' into a new data variable data_2

data_2 = data_1[data_1['Crime Type'] == 'Murder or Manslaughter']





# ploting Victims Sex Statistics with respect to Crime Type. Below graph shows that 77.39 % of the victims are Males.
ggplot(data_2, aes('Crime Type', fill = 'Victim Sex')) + geom_bar()




# ploting Perpetrator Sex Statistics with respect to Crime Type.
ggplot(data_2, aes('Crime Type', fill = 'Perpetrator Sex')) + geom_bar()




# Ploting Victim Race Statistic with respect to Crime Type. 
ggplot(data_2, aes('Crime Type', fill = 'Victim Race')) + geom_bar() 




# Ploting Perpetrator Race Statistic with respect to Crime Type.
ggplot(data_2, aes('Crime Type', fill = 'Perpetrator Race')) + geom_bar()




# Ploting Weapon Statistic with respect to Crime Type. 
# The below graph shows that Handgun alone is used in 50% of the crimes.  
# and Knife(Family member of blunt object) and Blunt Object()  is the 2nd most used Weapon.
ggplot(data_2, aes('Crime Type', fill = 'Weapon')) + geom_bar()




# On Creating the new family for weapons and puting the different wepons to there root family
# Lets analyse the above statstics in from different view.
data_3 = data_2
data_3.loc[data_3.Weapon == 'Knife', 'Weapon'] = "Blunt Object"
data_3.loc[data_3.Weapon == 'Gun', 'Weapon'] = "Firearm"
data_3.loc[data_3.Weapon == 'Handgun', 'Weapon'] = "Firearm"
data_3.loc[data_3.Weapon == 'Rifle', 'Weapon'] = "Firearm"
data_3.loc[data_3.Weapon == 'Shotgun', 'Weapon'] = "Firearm"
data_3.loc[data_3.Weapon == 'Rifle', 'Weapon'] = "Firearm"
data_3.loc[data_3.Weapon == 'Strangulation', 'Weapon'] = "Suffocation"
data_3.loc[data_3.Weapon == 'Drowning', 'Weapon'] = "Suffocation"

# Ploting Weapon Statistic with respect to Crime Type.  From the below graph we can make that 
# Firearm(Gun, Handgun, Rifle, Shotgun, Rifle) is the top source for Crime ( > 60 %)
# Followed by Blunt Objects (Blunt Objects, Knife) followed by Unknown

ggplot(data_2, aes('Crime Type', fill = 'Weapon')) + geom_bar()




# Below graph shows the distribution of crime solved with respect to the Year from 1980 till 2014
ggplot(data_2, aes('Year' ,'Crime Solved', fill = 'Crime Solved' )) + geom_bar(stat = "identity") + theme(axis_text_x  = element_text(angle = 90, hjust = 1))

