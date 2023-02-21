#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ggplot import *

#import seaborn as sns #visualization
#sns.set_style('whitegrid')




hdata = pd.read_csv('../input/database.csv')




#Preview the Data (Beginning)
pd.set_option('display.max_columns', None)
hdata.head(10)




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




#print(len(np.unique(hdata['Agency Name'])))
print((np.unique(data_1['Crime Type'])))
print((np.unique(data_1['Victim Race'])))
print((np.unique(data_1['Perpetrator Race'])))
print((np.unique(data_1['Relationship'])))




# ploting crime type
ggplot(data_1, aes('Crime Type')) + geom_bar()





# From the above plot we can see that 'Murder or Manslaughter' occurs more often than 'Manslaughter by Negligence.'
# therefor Extracting the data based on Crime type  = 'Murder or Manslaughter' into a new data variable data_2

data_2 = data_1[data_1['Crime Type'] == 'Murder or Manslaughter']




# ploting Victims Sex Statistics with respect to Crime Type. Below graph shows that 77.39 % of the victims are Males.
ggplot(data_2, aes('Crime Type', fill = 'Victim Sex')) + geom_bar()




# Calculating and mapping State with the count of crimes
#stat_homicide_count = data_2.groupby(["State"]).size().reset_index(name="Count").sort_values(by='Count', ascending=False)
#print stat_homicide_count

#Top 3 State with the highest Crime Type 'Murder or Manslaughter' and there representation as plot below.
# 1)             California  98994  (4)
# 2)             Texas       61087  (43)
# 3)             New York    49222  (32)

#ggplot(data_2, aes('State', fill = 'State')) + \
ggplot(data_2, aes('State')) + geom_bar() + theme(axis_text_x  = element_text(angle = 90, hjust = 1))

