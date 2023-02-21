#!/usr/bin/env python
# coding: utf-8



# Load library
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import csv as csv 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Get the raw data
train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)
full_df = pd.concat([train_df, test_df]) #Bind training and test data

#check data
# print(full_df.head(5))
print(full_df.info())




# Survival rate depending on passenger's class
print( full_df[["Pclass", "Survived"]].groupby(['Pclass'],as_index=False).mean() )
full_df['Class1'] = (full_df['Pclass'] == 1).astype(int)




full_df['Title'] = full_df['Name'].str.replace('(.*, )|(\\..*)', '') # creation of a new feature "Title" in the dataframe

# These line help me to find out a problem, there was Miss with male sex. 
# I correct the mistake but I let those lines to show how I found.
    #print (full_df.groupby(['Title','Sex']).size())
    #print(pd.crosstab(full_df['Title'],full_df['Sex']))
    #print( full_df[(full_df['Title'] == 'Miss') & (full_df['Sex'] == 'male')].head(3))

print(pd.crosstab(full_df['Title'],full_df['Sex']))




# We can see there is Title with few counts, we will group them into special title
full_df['Title'] = full_df['Title'].replace(['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'special_title')

# Also reassign mlle, ms, and mme accordingly
full_df['Title'] = full_df['Title'].replace('Mlle', 'Miss')
full_df['Title'] = full_df['Title'].replace('Ms', 'Miss')
full_df['Title'] = full_df['Title'].replace('Mme', 'Mrs')

# Relation between Title, sex and survival rate.
print( full_df[["Title","Sex", "Survived"]].groupby(['Title', 'Sex'],as_index=False).mean() )

#create a similar feature but a numeric feature for the learning algorihm
full_df['Title_Num'] = full_df['Title'].map( {'Mrs': 0, 'Miss': 1, 'special_title':2, 'Master':3, 'Mr':4} ).astype(int) 




print( full_df[["Sex", "Survived"]].groupby(['Sex'],as_index=False).mean() )
#transform Sex into numeric feature for the learning algorithm
full_df['Gender'] = full_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int) 




# Create a new feature: family size (including the passenger themselves)
full_df['FamilySize'] = full_df['SibSp'] + full_df['Parch'] + 1
cor_FSize = full_df[["FamilySize", "Survived"]].groupby(['FamilySize'],as_index=False).mean()
print(cor_FSize)
    #plt.bar(cor_FSize['FamilySize'], cor_FSize['Survived'])
    #plt.xlabel('Family Size')
    #plt.ylabel('Survival Rate')




full_df['FamilySizeCategories'] = full_df['FamilySize']
full_df.loc[ full_df['FamilySizeCategories'] == 1, 'FamilySizeCategories' ] = 0 #Singleton
full_df.loc[ (full_df['FamilySizeCategories'] > 1) & (full_df['FamilySizeCategories'] < 5) , 'FamilySizeCategories' ] = 1 #Small
full_df.loc[ full_df['FamilySizeCategories'] > 4, 'FamilySizeCategories' ] = 2 #Large
print( full_df[["FamilySizeCategories", "Survived"]].groupby(['FamilySizeCategories'],as_index=False).mean() )




# replace missing value of Fare
full_df.loc[ full_df['Fare'].isnull(), 'Fare' ] = full_df['Fare'].mean()
#transform Embarked into numeric value for the learning algorithm
full_df['Embarked_Num'] = full_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
# fill the 3 missing values
full_df.loc[ full_df['Embarked'].isnull(), 'Embarked_Num' ] = 1




# Keep track of what was he missing value
full_df['AgeIsNull'] = pd.isnull(full_df.Age).astype(int)

#Fill the missing value with the median value of people having same class and gender.
full_df['AgeFill'] = full_df['Age']
median_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = full_df[(full_df['Gender'] == i) &                               (full_df['Pclass'] == j+1)]['Age'].dropna().median()
        full_df.loc[ (full_df.Age.isnull()) & (full_df.Gender == i) & (full_df.Pclass == j+1),         'AgeFill'] = median_ages[i,j]

# plot old Age and new Age values
fig, axs = plt.subplots(1,2)
full_df['Age'][~np.isnan(full_df['Age'])].hist(ax=axs[0], bins=16)
full_df['AgeFill'].hist(ax=axs[1], bins=16)




# get average, std, and number of NaN values
average_age_titanic   = full_df["Age"].mean()
std_age_titanic       = full_df["Age"].std()
count_nan_age_titanic = full_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + 
                           std_age_titanic, size = count_nan_age_titanic)

# fill NaN values in Age column with random values generated
full_df.loc[np.isnan(full_df["Age"]), "AgeFill"] = rand_1

# plot old Age and new Age values
fig, axs = plt.subplots(1,2)
full_df['Age'][~np.isnan(full_df['Age'])].hist(ax=axs[0], bins=16)
full_df['AgeFill'].hist(ax=axs[1], bins=16)




Ahhh ! This method seems better because the repartition seems similar this time.
Let's create new feature with Age: Child.




full_df['Child'] = (full_df['Age'] < 18).astype(int)
print( full_df[["Child", "Sex", "Survived"]].groupby(['Child', 'Sex'],as_index=False).mean() )




full_df['Mother'] = ((full_df['Gender'] == 0) & (full_df['AgeFill'] > 18) & (full_df['Title'] == "Miss")).astype(int)
print( full_df[["Mother", "Survived"]].groupby(['Mother'],as_index=False).mean() )
print( full_df[["Sex", "Survived"]].groupby(['Sex'],as_index=False).mean() )




print (full_df.info() )




full_df = full_df.drop(['Age', 'Cabin', 'Embarked', 'Name', 'Sex', 'Ticket', 'Title'], axis=1)




# Create the train and test set for our algorithms
train_df = full_df[0:890]
test_df = full_df[891:1309]
X_train = train_df.drop(['Survived', 'PassengerId'],axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(['Survived', 'PassengerId'],axis=1).copy()




# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
forest = random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
#random_forest.score(X_train, Y_train)




# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)




coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(forest.feature_importances_)

# preview
coeff_df.sort_values(["Coefficient Estimate"], ascending=False)




output = Y_pred.astype(int)
ids = test_df['PassengerId'].values
predictions_file = open("titanic_predict.csv", "w") # Python 3
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()

