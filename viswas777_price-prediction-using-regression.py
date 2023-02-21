#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import sklearn




from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv('../input/autos.csv', encoding='latin_1')

# This it the data of cars listed for sale in ebay germany. The data has 371540 listings and 20 attributes.
Below is the brief description of all the attributes of data

dateCrawled         : when advert was first crawled, all field-values are taken from this date
name                : headline, which the owner of the car gave to the advert
seller              : 'privat'(ger)/'private'(en) or 'gewerblich'(ger)/'dealer'(en)
offerType           : 'Angebot'(ger)/'offer'(en) or 'Gesuch'(ger)/'request'(en)
price               : the price on the advert to sell the car
abtest              : ebay-intern variable (argumentation in discussion-section)
vehicleType         : one of eight vehicle-categories 
yearOfRegistration  : at which year the car was first registered
gearbox             : 'manuell'(ger)/'manual'(en) or 'automatik'(ger)/'automatic'(en)
powerPS             : the power of the car in PS
model               : the cars model
kilometer           : how many kilometres the car has driven
monthOfRegistration : at which month the car was first registered
fuelType            : one of seven fuel-categories
brand               : the cars brand
notRepairedDamage   : if the car has a damage which is not repaired yet
dateCreated         : the date for which the advert at 'ebay Kleinanzeigen' was created
nrOfPictures        : number of pictures in the advert
postalCode          : where in germany the car is located
lastSeenOnline      : when the crawler saw this advert last online


len(df)




df.head()




df.drop(['name','abtest','offerType','dateCrawled','nrOfPictures','seller','postalCode'],axis=1,inplace=True)




df.columns=['Price','Type','Registration_Year','Automatic','Power','Model','Kilometers','Registration_Month','Fuel','Brand','Repaired','Listed_Time','Sold_Time']




df.head()




df=df.ix[(df.Registration_Year>1950)& (df.Registration_Year<=2016)]




len(df)




#Replacing all the 0 month values to 1
df.Registration_Month.replace(0,1,inplace=True)
# Making the year and month colun to get a single date
Purchase_Datetime=pd.to_datetime(df.Registration_Year*10000+df.Registration_Month*100+1,format='%Y%m%d')
import time
from datetime import date
y=date(2017, 3,1)
# Calculating days old by subracting both date fields and converting them into integer
Days_old=(y-Purchase_Datetime)
Days_old=(Days_old / np.timedelta64(1, 'D')).astype(int)
type(Days_old[1])
df['Days_old']=Days_old




df.head()




df.drop(['Registration_Year','Registration_Month','Listed_Time','Sold_Time'],axis=1,inplace=True)




df=df.ix[(df.Power>=10)]




df.isnull().any()
   




df=df.ix[(df.Price>400)&(df.Price<=40000)& (df.Kilometers>1000)&(df.Kilometers<=150000),:]




df.replace({'Repaired': {'ja': 'Yes','nein': 'No',np.nan:'No'}},inplace=True)




# making nulls in Gearbox as  as Manual(0) 
df.replace({'Automatic': {'manuell': 'manual','automatik': 'automatic',np.nan:'manual'}},inplace=True)
# making nulls in repaired as not repaired (0)




df.Fuel.unique()
df.replace({'Fuel': {'benzin': 'benzene','andere': 'other','elektro':'electrical','lpg':'gas','cng':'gas',np.nan:'other'}},inplace=True)
df.Fuel.unique()




df.isnull().any()
   




df=pd.get_dummies(data=df,columns=['Type','Automatic','Model','Fuel','Brand','Repaired'],)




df.head()




X=df.drop('Price',axis=1)




Y1=df.Price




from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import linear_model

regs = [LinearRegression(), ElasticNet(), DecisionTreeRegressor(),DecisionTreeRegressor(max_depth=2),DecisionTreeRegressor(max_depth=5), GradientBoostingRegressor(),linear_model.Lasso()]
 




from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X,Y1,test_size=0.3,random_state=3)




maxScore1=0
for reg1 in regs:
    reg1.fit(X1_train,y1_train)
    thisScore1 = reg1.score(X1_test,y1_test)
    print (str(reg1) +': ' +str(thisScore1))
    if thisScore1 > maxScore1:
        bestScore1 = reg1
        maxScore1 = thisScore1
    




bestScore1




maxScore1

