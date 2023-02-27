#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from pandas import Series, DataFrame




titanic_df = pd.read_csv('train.csv')




titanic_df.head()









#Exporing the columns

titanic_df.info()









import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




# Let's first check gender
sns.factorplot('Sex', data=titanic_df,kind='count')




sns.factorplot('Sex', data=titanic_df,kind='count',hue='Pclass')




sns.factorplot('Pclass', data=titanic_df,kind='count',hue='Sex')




# This tells us that there was more males in the 3rd class compartment 




# First let's make a function to sort through the sex 
def male_female_child(passenger):
    # Take the Age and Sex
    age,sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex
    

# We'll define a new column called 'person', remember to specify axis=1 for columns and not index
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)




# Let's see if this worked, check out the first ten rows
titanic_df[0:10]




sns.factorplot('Pclass', data=titanic_df,kind='count',hue='person')




#Lets see the histogram of the Age column

titanic_df['Age'].hist(bins=70)




#Average Age on the Ship was ---
titanic_df['Age'].mean()




# Lets see how many women, men and children were on board
titanic_df['person'].value_counts()




fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest =  titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()




# Kernel density plot by male, female and child

fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest =  titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()




#Distribution by Class

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest =  titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()




titanic_df.head()




deck = titanic_df['Cabin'].dropna()




deck.head()









# So let's grab that letter for the deck level with a simple for loop

# Set empty list
levels = []

# Loop to grab first letter
for level in deck:
    levels.append(level[0])    

# Reset DataFrame and use factor plot
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,kind='count',palette='winter_d')




cabin_df = cabin_df[cabin_df.Cabin != 'T']




sns.factorplot('Cabin',data=cabin_df,kind='count',palette='summer')




titanic_df.head()




#Let's research about the ports embarked

sns.factorplot('Embarked',data=titanic_df,kind='count',hue='Pclass')




#Lets see how was the gender distributed

sns.factorplot('Embarked',data=titanic_df,kind='count',hue='person')




# Who was alone  and who was with family
titanic_df.head()




# Lets mine the SibSp and Parch to understand passenger who were alone




titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch




titanic_df.head()









titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'with family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'




titanic_df.head()




titanic_df[titanic_df['Alone'] == 'Alone'].count()




sns.factorplot('Alone',data=titanic_df,palette='Blues', kind='count')




#Lets see if they survived the fateful day
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})




titanic_df.head()




sns.factorplot('Survivor',data=titanic_df,palette='Set1',kind='count')




sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)




sns.lmplot('Age','Survived',titanic_df)




sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter')




generations = [10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)




sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)

