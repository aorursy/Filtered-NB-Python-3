#!/usr/bin/env python
# coding: utf-8



#import the necessary packages

#data wrangling
import numpy as np 
import pandas as pd
import random

#data visualisation
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#machine learning 
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC




#create a dataframe for the training data
#data file path obtained from the "add data" tab
train_data= pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")




#view the training data to make sure it is correct
train_data.head()




#Data analysis
#Make sure the datatypes are correct
#Think: Which columns are categorical? --> Survived, Pclass, Sex, embarked 
#Which are numerical? --> PassengerId, age, sibsp, parch, fare
#Which are mixed (e.g. alphanumerical)? These are good candidates for correction --> ticket, cabin
train_data.info()
print("-"*40)
test_data.info()

#Notice how some age, cabin and embarked entries are missing
#Remove or fill in blank/ null values




#Data analysis
train_data.describe()

#891 out of 2224 translates to 40% of the total population
#Survival rate of 38% is representative compared to the actual 32% survival rate provided by the competition description (1502 out of 2224 died)
#50% of the population are between the ages of 20 and 38
#75% travelled without parents and children
#50% travelled without siblings or spouse




#Data analysis
train_data.describe(include=['O'])

#note it's capital o, not 0. O stands for object. Note how previously describe only shows the numerical data. But here it shows the categorical data 




#Data analysis
#See whether there is a trend e.g if it's in ascending/descending order

train_data[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(["Survived"], ascending=False)




train_data[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(["Survived"],ascending=False)




#To see the sruvival rates within each class, further broken down by age
g=sns.FacetGrid(train_data, "Pclass", hue="Survived")
g.map(plt.hist,"Age").add_legend()




#Preliminary observation

#classify
#add the following assumptions based on competition description
    #women were more likely to survive
    #children (<?) were more likely to survive
    #upper-class passengers (pclass=1) were more likely to survive

#complete
#may want to complete the age field, as it is correlated with survival
#may also want to complete the embarked field

#drop
#note how 210 tickets are duplicated (22%) and likely not correlated with survival--> may want to drop this variable 
#cabin feature is highly incomplete (only 204 out of 891)--> may want to drop it
#passengerID likely not correlated with survival--> can be dropped
#name does not contribute to survival 

#engineer
#create a new feature called family based on how many sibsp + parch the person has on board
#create a new feature called title, to extract from name the honourific titles, dr , miss, madam etc.
#create a new feature called age bands --> turns a continuous numerical feature into an ordinal categorical feature
#maybe a fare range feature too, although this could be correlated with pclass?




#Engineer new features and adding it as a column in the dataframe




#Logistic regression (since the output is yes/no)




#Random forest




#Kernel support vector machine




#Output should have exactly 418 entries plus a header row, and 2 columns (PassengerID, Survived)

output= pd.DataFrame("PassengerID": test_data.PassengerId, "Survived": predictions)
output.to_csv("my_submission.csv", index= False)
print("Your submission is successfully saved!")

