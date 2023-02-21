#!/usr/bin/env python
# coding: utf-8



# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', "inline # it's mandatory to  visualize data")

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB




# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, na_values=['NA',''])
test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, na_values=['NA',''] )

# preview the data
titanic_df.head()




titanic_df.info()
print("----------------------------")
test_df.info()




# drop unnecessary columns, these columns won't be useful in analysis and prediction
# df = df.drop('column_name', 1), 0 for rows and 1 for columns 
#titanic_df.drop(['PassengerId','Name','Ticket','Cabin'],1)
#test_df.drop(['PassengerId','Name','Ticket','Cabin'],1)




# Plotting of Embarkment at different points and Survival
sb.factorplot(x="Embarked", data=titanic_df, kind="count",
              palette='BuPu', hue='Survived', size=6, aspect=1.5)




# Plotting Sex vs Survival
sb.factorplot(x="Sex", data=titanic_df, kind="count",
              palette="BuPu", hue='Survived',size=6, aspect=1.5)




# Try some functions
# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.
# "0::" means all (from start to end), and Python starts indices from 0 (not 1)
data = titanic_df 
nb_passengers = data['Survived'].count()
nb_survived = data.loc[data['Survived'] == 1, 'Survived'].count()
print(nb_survived / nb_passengers)

#titanic_df.head()
#titanic_df.tail()
#titanic_df.describe()
#print(titanic_df.groupby('Survived').size())
#print(titanic_df['Survived']).sum())




women_only_stats = data['Sex'] == "female" # This finds where all 
                                           # the elements in the gender
                                           # column that equals “female”
men_only_stats = data['Sex'] != "female"   # This finds where all the 
                                           # elements do not equal 
                                           # female (i.e. male)




#source: https://www.kaggle.com/c/titanic/details/getting-started-with-python-ii
# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv("../input/train.csv", header=0)
df.head(3)
type(df)
df.dtypes
df.info()
df.describe()




#df['Age'][0:10]
type(df['Age'])
print(df['Age'].mean())
print(df['Age'].median())




df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]




df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']] #177 missings
df[df['Embarked'].isnull()][['Embarked','Sex', 'Pclass', 'Age']] #2 missings




for j in range(1,4):
    print j, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == j) ])




import pylab as P
df['Age'].hist()
P.show()
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()




df['Gender'] = 4
df['Gender'] = df['Sex'].map( lambda x: x[0].upper() ) #x[0] of any string returns its first character
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

df['EmbarkedFill'] = df['Embarked'].fillna('S')
df[ df['Embarked'].isnull() ][['Gender','Pclass','Embarked','EmbarkedFill']].head(10)

df['Port'] = df['EmbarkedFill'].map( {'S': 1, 'C': 0, 'Q': 3} ).astype(int)




#Treat the missing values

median_ages = np.zeros((2,3)) #because 2 genders and 3 classes

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) &                               (df['Pclass'] == j+1)]['Age'].dropna().median()
median_ages




df['AgeFill'] = df['Age']

df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)




for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),                'AgeFill'] = median_ages[i,j]
        
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df




df.dtypes
#df.dtypes[df.dtypes.map(lambda x: x=='object')]




#df = df.drop(['PassengerId'], axis=1)
df.dtypes




#The final step is to convert it into a Numpy array
df =df[['Survived', 'AgeFill', 'Port', 'Gender']]
train_data = df.values
train_data




#CORRECTION DATA TEST
dft = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
dft['Gender'] = dft['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
dft['Port'] = dft['Embarked'].map( {'S': 1, 'C': 0, 'Q': 3} ).astype(int)
dft['AgeFill'] = dft['Age']
for i in range(0, 2):
    for j in range(0, 3):
        dft.loc[ (dft.Age.isnull()) & (dft.Gender == i) & (dft.Pclass == j+1),                'AgeFill'] = median_ages[i,j]
        
dft[ dft['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
dft['AgeIsNull'] = pd.isnull(dft.Age).astype(int)
dft =dft[['AgeFill', 'Port','Gender']]
#df Survived Pclass SibSp Parch Fare Gender AgeFill AgeIsNull
test_data = dft.values
test_data




# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)
output




#Numerical values in test_data
numeric_variables_test = list(test_data.dtypes[test_data.dtypes != "object"].index)
test_data[numeric_variables_test].head()

logreg = LogisticRegression()
X_train = train_data[numeric_variables]
test_data.drop(["PassengerId"],axis=1)
X_test = test_data[numeric_variables]
y_test = y_train

#Fit the model with X_train and y_train
logreg.fit(X_train,y_train)

#Predict the response values for the observations in X_train
#y_prediction = logreg.predict(X_train)
y_prediction = logreg.predict(X_test)


#Check how many predictions were generated
len(y_prediction)
print(len(y_prediction))
print(y_prediction)


#Compute the score for the logistic regression model
logreg.score(X_train,y_train)

