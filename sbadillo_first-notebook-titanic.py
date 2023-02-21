#!/usr/bin/env python
# coding: utf-8



## importing libraries: 
## data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

## visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

## machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC # Scalable Support Vector for Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier  # stochastic gradient descent
from sklearn.tree import DecisionTreeClassifier




# importing data set as dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

print('length of train_df:', len(train_df), 'entries')
print('length of test_df: ', len(test_df), 'entries')




# preview the first rows of our training set.
train_df.head(6)




# info for each dataframe
for df in combine: 
    print('-'*20)
    df.info()




# plot some categorical features
# get distribution of port of embarkation
grid = sns.FacetGrid(train_df, hue='Embarked')
grid.map(plt.hist, 'Age', alpha=0.5)
grid.add_legend();

grid = sns.FacetGrid(train_df, hue='Sex')
grid.map(plt.hist, 'Age', alpha=0.5)
grid.add_legend();

grid = sns.FacetGrid(train_df, hue='Pclass')
grid.map(plt.hist, 'Age', alpha=0.5)
grid.add_legend();




# get summary statistics for numerical features
# play with percentages by using option (percentiles=[.8.,9])

train_df.describe(percentiles=[.1,.2,.25,.3,.4,.5,.6,.77,.8,.9,.95, .98,.99])




# get summary from categorical columns
train_df.describe(include=['O'])   # include=['O'] to call for categoricals




train_df[['Sex', 'Survived']].groupby(['Sex'], 
as_index=False).mean().sort_values(by='Survived', ascending=False)




train_df[['Pclass', 'Survived']].groupby(['Pclass'], 
as_index=False).mean().sort_values(by='Survived', ascending=False)




train_df[['SibSp', 'Survived']].groupby(['SibSp'], 
as_index=False).mean().sort_values(by='Survived', ascending=False)




train_df[['Parch', 'Survived']].groupby(['Parch'], 
as_index=False).mean().sort_values(by='Survived', ascending=False)




g = sns.FacetGrid(train_df, hue='Survived');
g.map(plt.hist, 'Age', bins=20) ;




# create plot grid.
# Showing Survival as separate hues, splitting by Pclass categories; vs Age
grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived', palette='Set1')
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# create plot grid.
# Showing Survival as separate hues, splitting by Sex; vs Age
grid = sns.FacetGrid(train_df, col='Sex', hue='Survived', palette='Set1') 
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();




# plot categorical features
grid = sns.FacetGrid(train_df, col='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=['blue','violet'])
grid.add_legend();




# Combining categorical values

#barplots: separated by port of embarkation, showing Fare, Sex and Survival.
grid = sns.FacetGrid(train_df, col='Embarked')
grid.map(sns.barplot, 'Sex', 'Fare', hue='Survived', data=train_df, alpha=.5, palette='Set1')
grid.add_legend(title='Survived');




train_df.head()




#### Dropping Cabin columns as we are not gonna use them.
print("dataframes before shape:", train_df.shape, test_df.shape)

train_df = train_df.drop(['Cabin'], axis= 1) 
test_df = test_df.drop(['Cabin'], axis=1)
combine = [train_df, test_df]

print("dataframes After shape:", train_df.shape, test_df.shape)




for df in combine:
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)
    
print('--'*10)
print(pd.crosstab(train_df['Title'], train_df['Sex']))
print('--'*10)
print(pd.crosstab(test_df['Title'], test_df['Sex']))




# grouping together some rare titles
for df in combine:
    df['Title'] = df['Title'].replace('Mlle','Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace('Don', 'Mr')
    
    rare_titles = ['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer',                    'Lady', 'Major', 'Rev', 'Sir', 'Dona']
    
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')




train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by="Survived", ascending=False)




#Convert some categorical features to numerical values.
title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
sex_mapping = {"female":0, "male":1}

for df in combine:
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    df['Sex'] = df['Sex'].map(sex_mapping).astype(int)
    
#drop Name columns
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]




train_df.head()




grid = sns.FacetGrid(train_df, col='Pclass', hue='Sex', palette=['blue', 'violet'])
grid.map(plt.hist, 'Age', alpha=0.5)
grid.add_legend();




guess_ages = np.zeros((2,3)) ; print(guess_ages)




for df in combine:
    print('---'*8)   

    for i in range(0, 2):
        for j in range (0,3):
            # filter the dataframe for only the categories we want...
            # create a new dataframe
            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j+1)]['Age'].dropna()
            
            # guess using median age
            age_guess = guess_df.median()
            # round to the nearest .5 and put value un matrix
            guess_ages[i,j] = int(.5 * round(float(age_guess)/.5)) 
            ()
    print(guess_ages)
    
    # search for missing Age values and substitute
    for i in range(0,2):
        for j in range(0,3):
            df.loc[ (df.Age.isnull()) & (df['Sex'] == i) & (df['Pclass'] == j+1) , 'Age'] = guess_ages[i,j]
    
    # convert float to integer
    df['Age'] = df['Age'].astype(int)




train_df['AgeBand'] = pd.cut(train_df['Age'],5)
train_df[['Survived','AgeBand']].groupby(['AgeBand']).mean()




# changing Age column to their corresponding Age Band ordinal value.
# do it for both dataframes
for df in combine:
    df.loc[ df['Age'] <= 16 , 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[(df['Age'] > 64), 'Age'] = 4




train_df.head()




# dropping AgeBand feature
train_df = train_df.drop(["AgeBand"], axis=1)
combine = [train_df, test_df]
train_df.head()




# create new feature of SharingTicket that describes the amount of passengers sharing the Ticket
# then divide the Fare by SharingTicket

for df in combine:
    
    # count how many PassengerId's exists per each ticket number
    sharedTickets = df.groupby(['Ticket']).count()
    sharedTickets = sharedTickets.PassengerId
    print('found', sharedTickets[sharedTickets > 1].size, ' sharing ticket codes \n')
    print('| SharingSize | no. of entriesfound |\n', sharedTickets.value_counts())
    
    # Insert new feature "TicketShare" to signal the size of the group sharing the ticket
    df['TicketShare'] = df['Ticket'].map(sharedTickets)
    
    # Split original fare by number of the size of the group sharing it.
    # New feature called Fare per Person 'FarePP'
    df['FarePP'] = df.Fare / df.TicketShare  




# Drop TicketShare and Ticket column
# train_df = train_df.drop(['TicketShare', 'Ticket'], axis=1)
# test_df = test_df.drop(['TicketShare', 'Ticket'], axis=1)
# combine = [train_df, test_df]




# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

sns.boxplot(x="Pclass", y="Fare", data=train_df, ax=ax1)
sns.boxplot(x="Pclass", y="FarePP", data=train_df, ax=ax2);




# using median Fare and FarePP to complete the single missing entry in our test_df

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df['FarePP'].fillna(test_df['FarePP'].dropna().median(), inplace=True)




test_df[test_df['FarePP'].isnull()]




# creating Fare bands using qcut (Quantile based)

train_df['FareBand'] = pd.qcut(train_df['Fare'],5)
train_df[['Survived', 'FareBand']].groupby(['FareBand']).mean()




for df in combine:
    df.loc[(df['Fare'] <= 7.854) , 'Fare'] = 0
    df.loc[(df['Fare'] > 7.854)  & (df['Fare'] <= 10.5), 'Fare'] = 1
    df.loc[(df['Fare'] > 10.5 )  & (df['Fare'] <= 21.679), 'Fare'] = 2
    df.loc[(df['Fare'] > 21.679) & (df['Fare'] <= 39.688), 'Fare'] = 3
    df.loc[(df['Fare'] > 39.512), 'Fare'] = 4    




train_df['FarePPBand'] = pd.qcut(train_df['FarePP'],5)
train_df[['Survived', 'FarePPBand']].groupby(['FarePPBand']).mean()




for df in combine:
    df.loc[(df['FarePP'] <= 7.733) , 'FarePP'] = 0
    df.loc[(df['FarePP'] > 7.733)  & (df['FarePP'] <= 8.05), 'FarePP'] = 1
    df.loc[(df['FarePP'] > 8.05)  & (df['FarePP'] <= 11.725), 'FarePP'] = 2
    df.loc[(df['FarePP'] > 11.725) & (df['FarePP'] <= 26.55), 'FarePP'] = 3
    df.loc[(df['FarePP'] > 26.55), 'FarePP'] = 4 




train_df.head()




#dropping FareBand and FarePPBand
train_df = train_df.drop(['FareBand', 'FarePPBand'], axis=1)
combine = [train_df, test_df]
train_df.head()




#Dropping Ticket columns
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

combine = [train_df, test_df]
train_df.head()




#Dropping Ticketshare column
train_df = train_df.drop(['TicketShare'], axis=1)
test_df = test_df.drop(['TicketShare'], axis=1)

combine = [train_df, test_df]
train_df.head()




print(train_df[['Fare', 'Survived']].groupby(['Fare']).mean().sort_values(by="Survived", ascending=False))
print(train_df[['FarePP', 'Survived']].groupby(['FarePP']).mean().sort_values(by="Survived", ascending=False))




train_df[train_df.Embarked.isnull()]




# print(train_df.Embarked[train_df.Pclass == 1].mode()[0])
# print(train_df.Embarked[train_df.Fare == 4].mode()[0])
print('most common port =', train_df.Embarked.mode()[0])




# fill with most common value
train_df['Embarked'] = train_df['Embarked'].fillna(train_df.Embarked.mode()[0])

# transform to numerical
for df in combine:
    df['Embarked'] = df['Embarked'].map({"S":1,"Q":2,"C":3})

combine=[train_df,test_df]




train_df.head()




# Create new feature using family imformation in Parch and SibSp
for df in combine:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1 

train_df.head()




print(train_df[['FamilySize', 'Survived']].groupby('FamilySize').mean().sort_values(by="Survived"));
print(train_df[['IsAlone', 'Survived']].groupby('IsAlone').mean())




for df in combine:
    df.loc[(df['FamilySize'] >= 3) & (df['FamilySize'] < 5), 'FamilySize'] = 3
    df.loc[(df['FamilySize'] >= 5), 'FamilySize'] = 4




train_df[['FamilySize', 'Survived']].groupby('FamilySize').mean().sort_values(by='Survived', ascending=False)




#Dropping PassengerID column
train_df = train_df.drop(['PassengerId'], axis=1)

combine = [train_df, test_df]
train_df.head()




X_train = train_df.drop(["Survived"], axis=1)
Y_train = train_df["Survived"]

# for the test_df we will need the passengerId later, so we will subit a copy of the dataframe
X_test  = test_df.drop(["PassengerId"], axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape




# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log




coeffs = pd.DataFrame(train_df.columns.drop('Survived'))
coeffs.columns = ['Feature']
coeffs['Correlation'] = pd.Series(logreg.coef_[0])
coeffs['squared'] = coeffs['Correlation'] ** 2




coeffs.sort_values(by= 'Correlation')




sns.barplot(y='Feature', x='Correlation', data=coeffs, order= coeffs.sort_values(by= 'Correlation').Feature );




sns.barplot(y='Feature', x='squared', data=coeffs, order= coeffs.sort_values(by= 'squared').Feature );




# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc




knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn




# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian




# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc




# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd




# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)


Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest




Importance = pd.DataFrame(train_df.columns.drop('Survived'))
Importance.columns = ['Feature']
Importance['Importance'] = pd.Series(random_forest.feature_importances_)




ax = sns.barplot(data=Importance, y='Feature', x='Importance', order=(Importance.sort_values('Importance', ascending=False).Feature))




models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, 
              acc_sgd, acc_linear_svc]})
models.sort_values(by='Score', ascending=False)




plot = sns.barplot(y=models.Model, x=models.Score, order=(models.sort_values(by='Score', ascending=False).Model),)

# Adding the score vvalue as labels.
for p in plot.patches:
    width = p.get_width()
    plt.text(width, (p.get_height()/1.5 + p.get_y()), width,
            ha="right") 




# Using Y_pred from our last model Random Forests to create submission
# submission should contain PassengerId and its predicted survival value.

submission = pd.DataFrame({
    'PassengerId':test_df['PassengerId'],
    'Survived':Y_pred
})

