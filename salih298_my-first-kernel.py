#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




data = pd.read_csv('../input/2017.csv')




data.info()




data.columns=['Country','Happiness_Rank', 'Happiness_Score', 'Whisker_high',
       'Whisker_low', 'Economy_GDP_per_Capita', 'Family',
       'Health_Life_Expectancy', 'Freedom', 'Generosity',
       'Trust_Government_Corruption', 'Dystopia_Residual']
data.corr()




# correlation
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot= True,linewidths= 5, fmt='.1f',ax=ax)
plt.show()




data.Family.plot(kind='line',color='g',grid=True,label='Family',alpha=0.5,linewidth=1,linestyle=':')
data.Freedom.plot(color='r',grid=True,alpha=0.5,label='Freedom',linewidth=1,linestyle='-')  
plt.legend()
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot ')
plt.show()




Scatter Plots




plt.scatter(data.Family,data.Freedom,color='b',alpha=0.5)
plt.xlabel('Family')
plt.ylabel('Freedom')
plt.title('Family Freedom Scatter Plot')
plt.show()




data.plot(kind='scatter',x='Family',y='Freedom',color='r',alpha=0.5)

plt.show()




data.columns




data.Health_Life_Expectancy.plot(kind='hist',bins=50,figsize=(12,12))
plt.show()




data.Health_Life_Expectancy.plot(kind='hist',bins=50,figsize=(12,12))
plt.clf() # clears the method




dictionary = {'galatasaray':'taffarel','fenerbahçe':'volkan','beşiktaş':'rüştü','trabzonspor':'şenol'}
# print(dictionary)
print(dictionary.keys())
print(dictionary.values())




# changing the value
dictionary['galatasaray']='muslera'
print(dictionary)
# adding a new entry
dictionary['juventus']='ronaldo'
print(dictionary)
# remove entry
#del dictionary['juventus']
print(dictionary)
print('trabzonspor' in dictionary)
#dictionary.clear
#print(dictionary)









data = pd.read_csv('../input/2017.csv')




series = data['Economy..GDP.per.Capita.']
print(type(series))
df = data[['Economy..GDP.per.Capita.']]
print(type(df))




# comparison operator
print(3 > 1)
print(3!=1)
# Boolean operators
print(True and False)
print(True and False and True and True and True and True)
print(True or False)
print(True and False and True and False or True)




# 1-)Filtering pandas data frame
x = data['Family']>1.3
data[x]




# 2-) Filtering pandas with logical_and or logical_or
data[np.logical_or(data['Family']>1.3,data['Happiness.Score']>7)]




# also could be written as :
data[(data['Family']>1.5) | (data['Happiness.Score']>7)]




i = 0
while i !=10:
    print('i is :',i)
    i = i + 1
print(i,' is equal to 10')




liste = [1,2,3,4,5,6,7,8,9,10]
for a in liste:
    print('a is :',a)
print('')

for index,values in enumerate(liste):
    print(index,':',values)
print('')

dictionary ={'gs':'muslera','fb':'volkan','bjk':'fabri'}
print(dictionary)
for keys,values in dictionary.items():
    print(keys,':',values)
print('')
for index,values in data[['Family']][75:101].iterrows():
    print(index,':',values)




def tuble_ex():
    """return defined t tuble"""
    t = (1,2,3)
    return t
a,b,c = tuble_ex()
print(a,b,c)




x = 5  # global scope
def f():
    x = 3   # local scope
    return x
print(x)      # x = 5 global scope
print(f())    # x = 3 local scope




# if there is no local scope
# it uses global scope x
x = 5
def f():
    y = x*2
    return y
print(f())




# if both does not exist built in scope is seached
import builtins
dir(builtins)




# nested function
def square():
    """ return square of value"""
    def add():
        """ add two local variable"""
        x = 2
        y = 3
        z = x+y
        return z
    return add()**2
print(square())




# default arguments
def f(a,b=1,c=2):
    y = a + b +c
    return y
print(f(5))
# what if we want to change default arguments
f(5,4,3)




# flexible arguments *args
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    """print key and value of dictionary"""
    for keys,values in kwargs.items():
        print(keys,':',values)
f(country='spain',capital='madrid',population=123456)




# lambda function
square = lambda x: x**2  
print(square(4))
tot = lambda x,y,z : x+y+z
print(tot(1,2,3))




number_list = [5,7,3,865,55]
y = map(lambda x: x**3,number_list)
print(list(y))




# Example of list comprehension
num1 = [1,2,3]
num2 = [i**6+30 for i in num1]
print(num2)




# Conditionals on iterable
num1 = [3,13,21,65]
num2 = [i*18 if i<13 else i**2 if i>=13 and i<22 else i**(1/2) for i in num1]
print(num2)




mean_1 = sum(data.Health_Life_Expectancy)/(len(data.Health_Life_Expectancy))
print(mean_1)
data['Health_Life_Expectancy_Level'] = ['high' if i >= mean_1 else 'low' for i in data.Health_Life_Expectancy] 
data.loc[:200,['Health_Life_Expectancy_Level','Health_Life_Expectancy']]




data = pd.read_csv('../input/2017.csv')
data.head()




data.columns = ['Country','Happiness.Rank','Happiness.Score','Whisker.high','Whisker.low','Economy.GDP.per.Capita','Family','Health.Life.Expectancy','Freedom','Generosity','Trust.Government.Corruption','Dystopia.Residual']
data.columns




data.tail(10)




data.shape




data.info()




data.describe()




print(data['Happiness.Score'].value_counts(dropna=False)) # if there is non values that is also be counted




data.describe()




data_new = data.head()
data_new




# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
data_new = data.tail(10)
melted = pd.melt(frame=data_new,id_vars='Country',value_vars=['Whisker.high','Whisker.low'])
melted




# Index is country
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index='Country',columns='variable',values='value')




data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row




data1 = data['Health.Life.Expectancy'].head()
data2 = data['Trust.Government.Corruption'].head()
conc_data_col = pd.concat([data1,data2],axis=1)
conc_data_col




data.dtypes









data.info()




data['Family'].value_counts(dropna=False)




data1 = data
data1['Family'].dropna(inplace=True)




assert 1==1




data["Family"].fillna('empty',inplace = True)




assert  data['Family'].notnull().all()




### dataframes from dictionary
country = ['Turkey','Azerbeycan','Germany']
capital = ['Ankara','Bakü','Berlin']
liste_row = ['Country','Capital']
liste_col = [country,capital]
zipped = list(zip(liste_row,liste_col))
dictionary = dict(zipped)
df = pd.DataFrame(dictionary)
df




# add new columns
df['Population'] = [200,100,150]
df




# broadcasting
df['Income']=0
df




# plotting all data
data1 = data.loc[:,['Family','Health.Life.Expectancy','Generosity']]
data1.plot()
plt.show()




# subplots
data1.plot(subplots=True)
plt.show()




# scatter plot
data1.plot(kind='scatter',x='Family',y='Generosity')
plt.show()




# hist plot
data1.plot(kind='hist',y='Family',bins=50,range=(0,155),normed=True)
plt.show()




# histogram subplot with non cumulative and cumulative 
fig,axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind='hist',y='Family',bins=50,range=(0,100),normed=True,ax=axes[0])
data1.plot(kind = "hist",y = "Family",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt.savefig('graph.png')
plt




data.describe()




time_list = ['1992-03-08','1992-04-12']
print(type(time_list[1])) # as we can see date is string
# however we want it to be datetime object
datatime_object = pd.to_datetime(time_list)
print(type(datatime_object))




# close warning
import warnings
warnings.filterwarnings('ignore')
# lets practise
data2 = data.head()
date_list = ['1992-01-10','1992-02-10','1992-03-10','1993-03-15','1993-03-16']
date_datetime = pd.to_datetime(date_list)
data2['data'] = date_datetime
data2 = data2.set_index('data')
data2




# Now we can select according to our date index
print(data2.loc['1993-03-16'])
print(data2.loc['1992-03-10':'1993-03-16'])




# We will use data2 that we create at previous part
data2.resample('A').mean() # resample about years




data2.resample('M').mean() # resample about months
# As you can see there are a lot of nan because data2 does not include all months




# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data2.resample('M').first().interpolate('linear')




# Or we can interpolate with mean()
data2.resample("M").mean().interpolate("linear")




# read data
data = pd.read_csv('../input/2017.csv')
data = data.set_index('Country')
data.head()




# indexing using square brackets
data['Health..Life.Expectancy.'][153]




# using column attribute and row label
data.Family[0]




# using loc accessor
data.loc['Netherlands','Family']




# Selecting only some columns
data[['Whisker.high','Whisker.low']]




# Difference between selecting columns: series and dataframes
print(type(data['Freedom'])) # series
print(type(data[['Freedom','Trust..Government.Corruption.']])) # DataFrame




# Slicing and indexing series
data.loc['Australia':'Chile','Family':'Freedom']




data.loc['Chile':'Australia':-1,'Family':'Freedom']




# From something to end
data.loc['Mali','Whisker.high':]




# Creating boolean series
boolean = data.Family>1.5
data[boolean]




# Combining filters
first_filter = data['Economy..GDP.per.Capita.']>1.3
second_filter = data['Health..Life.Expectancy.']>0.85
data[first_filter & second_filter]




# Filtering column based others
data.Family[data.Generosity>0.5]




# Plain python functions
def sum(x):
    return x+5
data['Whisker.high'].apply(sum)
    




# Or we can use lambda function
data['Whisker.high'].apply(lambda x:x+5)




# Defining column using other columns
data['total_power'] = data['Happiness.Score']*data['Freedom']
data.head()




# our index name is this:
print(data.index.name)
# lets change it
data.index.name = 'Countries'
data.head()




# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/2017.csv')
data.head()
# As you can see there is index. However we want to set one or more column to be index




# Setting index : Country is outer Happiness.Rank is inner index
data1 = data.set_index(['Country','Happiness.Rank'])
data1.head(100)
# data1.loc["Iceland","Family"] # how to use indexes




dic = {'treatment':['A','A','B','B'],'gender':['F','M','F','M'],'response':[10,45,5,9],'age':[15,4,72,65]}
df = pd.DataFrame(dic)
df




# pivoting
df.pivot(index='treatment',columns='gender',values='response')




df1 = df.set_index(['treatment','gender'])
df1
# lets unstack it




# level determines indexes
df1.unstack(level=0)




df1.unstack(level=1)




# change inner and outer level index position
df2 = df1.swaplevel(0,1)
df2




df




# df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(frame=df,id_vars='treatment',value_vars=['response','age'])




df




# according to treatment take means of other features
df.groupby("treatment").mean()   # mean is aggregation / reduction method
# there are other methods like sum, std,max or min




# we can only choose one of the feature
df.groupby("treatment").age.max() 




# Or we can choose multiple features
df.groupby("treatment")[["age","response"]].min() 




df.info()

