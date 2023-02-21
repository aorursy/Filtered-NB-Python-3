#!/usr/bin/env python
# coding: utf-8



"""
TITANIC DATA VARIABLE DESCRIPTION:

VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.

"""




## lOAD DATA TO PANDA DF

import pandas as pd
import numpy as np
titanic = pd.read_csv('../input/train.csv')




titanic.head(5)




## Create a correlation function to find the corr between different variables


def correlation(x, y):
    #correlation = average of (x in standard units) times (y in standard units)
        
    x_std = x.std(ddof=0)   ## ddof = 0 is the corrected bessel standard deviation.
    x_mean = x.mean()
    y_std = y.std(ddof=0)
    y_mean = y.mean()
    x_in_std = (x-x_mean)/x_std
    y_in_std = (y-y_mean)/y_std
    correlation = (x_in_std * y_in_std).mean()
    
    return correlation




# Create a dataset of people who survived and a data set of people who did not survive

titanic_survived = titanic[titanic.Survived == 1]
titanic_not_survived = titanic[titanic.Survived == 0]




# 1. Find Survival summary of Pclass
survived =  titanic['Survived']
pclass = titanic['Pclass']

print ("SURVIVED AND PCLASS \n")
# Use groupby to summarize the survival rate of Pclass and total in each group
pclassSurvival = titanic.groupby('Pclass')['Survived'].agg(['sum','count'])
pclassSurvival['percent'] = pclassSurvival['sum']/pclassSurvival['count']
print (pclassSurvival)
print ("")

# 2. Find the Survival summary of Sex
sex = titanic['Sex']

print ("SURVIVED AND SEX \n")
# Use groupby to summarize the survival rate of Sex and count in each group
sexSurvival = titanic.groupby('Sex')['Survived'].agg(['sum','count'])
sexSurvival['percent'] = sexSurvival['sum']/sexSurvival['count']
print (sexSurvival)
print ("")

# 3. Find the Survival summary of Age
sex = titanic['Age']

print ("SURVIVED AND AGE \n")

# Create summary statistics of Age, Age of survived and Age of non-survived

valid_all_titanic = titanic.dropna(subset = ['Age'])
valid_survived = titanic_survived.dropna(subset = ['Age'])
valid_not_survived = titanic_not_survived.dropna(subset = ['Age'])

print ("Summary of all passenger Age \n")
print (valid_all_titanic['Age'].describe())
print ("")
print ("Summary of Survived passenger Age \n")
print (valid_survived['Age'].describe())
print ("")
print ("Summary of non survived passenger Age \n")
print (valid_not_survived['Age'].describe())
print ("")




# Create a histogram of people who survived with age on x axis 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy

bins = numpy.linspace(0, 100, 20)

plt.hist(valid_survived['Age'], bins, alpha = 0.5, label = "Survived")
plt.legend(loc='upper right')
plt.show()

plt.hist(valid_not_survived['Age'], bins, alpha = 0.5, label = "Not Survived")
plt.legend(loc='upper right')
plt.show()




# Looking at the graph, We can establish that some age group have higher chances to survive. Break down the data further 
# by creating a new categorical variable


# Add new categorical variable to the data frame.
valid_all_titanic['Age-Group'] = np.where(valid_all_titanic['Age'] < 10, '0-10', 'other')
valid_all_titanic['Age-Group'] = np.where(valid_all_titanic['Age'] >= 10, '10-20', valid_all_titanic['Age-Group'])
valid_all_titanic['Age-Group'] = np.where(valid_all_titanic['Age'] >= 20, '20-30', valid_all_titanic['Age-Group'])
valid_all_titanic['Age-Group'] = np.where(valid_all_titanic['Age'] >= 30, '30-40', valid_all_titanic['Age-Group'])
valid_all_titanic['Age-Group'] = np.where(valid_all_titanic['Age'] >= 40, '40-50', valid_all_titanic['Age-Group'])
valid_all_titanic['Age-Group'] = np.where(valid_all_titanic['Age'] >= 50, '50-60', valid_all_titanic['Age-Group'])
valid_all_titanic['Age-Group'] = np.where(valid_all_titanic['Age'] >= 60, '60-70', valid_all_titanic['Age-Group'])
valid_all_titanic['Age-Group'] = np.where(valid_all_titanic['Age'] >= 70, '70 Above', valid_all_titanic['Age-Group'])

# Find the survival rate of specific age groups

print ("SURVIVED AND AGE GROUP \n")
# Use groupby to summarize the survival rate of Age group and total in each group

AgeGroupSurvival =  valid_all_titanic.groupby('Age-Group')['Survived'].agg(['sum','count'])
AgeGroupSurvival['percent'] = AgeGroupSurvival['sum']/AgeGroupSurvival['count']
print (AgeGroupSurvival)
print ("")




print ("SURVIVED AND SIBSP \n")
# Use groupby to summarize the survival rate of SibSp and total in each group

sibSpSurvival =  titanic.groupby('SibSp')['Survived'].agg(['sum','count'])
sibSpSurvival['percent'] = sibSpSurvival['sum']/sibSpSurvival['count']
print (sibSpSurvival)
print ("")
print ("SURVIVED AND PARCH \n")
# Use groupby to summarize the survival rate of Parch and total in each group

parchSurvival =  titanic.groupby('Parch')['Survived'].agg(['sum','count'])
parchSurvival['percent'] = parchSurvival['sum']/parchSurvival['count']
print (parchSurvival)
print ("")
print ("SURVIVED AND EMBARKED STATION \n")
# Use groupby to summarize the survival rate of Embarked and total in each group

EmbarkedSurvival =  titanic.groupby('Embarked')['Survived'].agg(['sum','count'])
EmbarkedSurvival['percent'] = EmbarkedSurvival['sum']/EmbarkedSurvival['count']
print (EmbarkedSurvival)
print ("")




print ("COUNT OF PASSENGER CLASS FROM EACH EMBARKED STATTION")
EmbarkedSurvival_1 =  titanic.groupby(['Embarked','Pclass'])['Survived'].agg(['count'])
print (EmbarkedSurvival_1)




print ("COUNT OF PASSENGER Parch Variable FROM EACH EMBARKED STATTION")
EmbarkedSurvival_4 =  titanic.groupby(['Embarked','Parch'])['Survived'].agg(['count'])
print (EmbarkedSurvival_4)




print ("AGE DISTRIBUTION OF PASSENGERS FROM VARIOUS STATIONS")

EmbarkedSurvival_3 =  valid_all_titanic.groupby(['Embarked','Age-Group'])['Survived'].agg(['count'])
print (EmbarkedSurvival_3)




WHAT MIGHT HAVE CAUSED A HIGHER RATE OF SURVIVAL FOR CHERBOURG STATION
-----------------------------------------------------------------------

Looking at above two summaries of data we can say that, percentage of first class passengers boarded from Cherbourg is very high compared to other stations.

Percentage of first class passenger boarded from different stations.

Cherbourg   = 85/(85+17+66) = 50.6%
Queenstown  = 2/(2+3+72) = .03 %
Southampton = 127/(127+164+353) = 19.7%

If we look at the female passenger data, we can see Cherbourg and Queenstown have similar distribution, but Southampton had much less female passengers than other stations. This higher % of females in Cherbourg might have contribiuted to a higher survival rate.

Percentage of female passengers boarded from different stations.

Cherbourg   = 73/(73+95) = 43.4%
Queenstown  = 36/(36+41) = 46%
Southampton = 203/(203+441) = 31.5% 

WHAT MIGHT NOT HAVE CAUSED A HIGHER RATE OF SURVIVAL FOR CHERBOURGH STATION
-----------------------------------------------------------------------------

Percentage of child passegers is not higher in Cherbourg. So this mnight not have contributed.

Percentage of child passengers boarded from different stations.

Cherbourg   = 9/130 = 6.9%
Queenstown  = 4/28 = 14.2%
Southampton = 49/554 = 8.8% 

We can make similar conclusion about Parch also. For Cherbourg and Southampton, we can see similar % of people whih Parch value of zero.

