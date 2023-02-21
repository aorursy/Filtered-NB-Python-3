#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')




#checking number of null values in each columns
print(train_df.isnull().sum())
print('--'*10)
print(test_df.isnull().sum())
#They're several missing values in columns, so we must look after of it.




train_df.head(1)




# Title created, for filling the missing 'Age'
# we make function that can make the title from name can standalone, and make new columns called 'Title'
# this function is used for deleting the first 'word' from column 'name', like index 0 at example above
# then name is Braund, so we deleting Braud first to get `Mr.` for title, and ignoring the rest.

def hapusD(kalimat,kata): 
    pos = kalimat.find(kata)
    #print(kata,pos)
    if pos!=-1:
        pjg = len(kata)
        kalimat = kalimat[pos+1:]
    return kalimat

#then for the second function we deleting the all string after we found the title Mr.
def hapusT(kalimat,kata):
    pos = kalimat.find(kata)
    #print(pos)
    if pos!=-1:
        kalimat = kalimat[:pos+len(kata)]
        #print(kalimat)
    return kalimat

#this function is for use another function, function hapusD and hapusT.
def hapusName(desc):
    desc = hapusD(desc,', ')
    desc = hapusT(desc,'. ')
    
    return desc.strip()




#applying function to Name for both train and test dataframe then create the new columns, Title.
train_df['Title'] = train_df['Name'].apply(hapusName)
test_df['Title'] = test_df['Name'].apply(hapusName)




#we want to know what title of each person had which has missing value.
print(train_df.loc[train_df['Age'].isnull(),'Title'].value_counts())
#there's almost common Title that has missing values.




#we want to plotting the mean distribution and deviation for each title
avg_age_title = train_df[['Title','Age']].loc[train_df['Age'].notnull()].groupby('Title').mean()
std_age_title = train_df[['Title','Age']].loc[train_df['Age'].notnull()].groupby('Title').std().fillna(0)

avg_age_title.index.names = std_age_title.index.names = [""]
fig, ax = plt.subplots(figsize = (15.7,6.27))
avg_age_title.plot(yerr=std_age_title,kind='bar',ax = ax, title = "Age and Deviation In each Title")
# so we will input missing values with random value around mean and deviation for each Title.




#so this is the function, we gather sex and the title together
def fillAge(inputs):    
    sex,title = inputs

    try:
        #if the title is known in by the training set, we random number around mean with deviation for each title.
        avg = train_df.loc[(train_df['Title'] == title),'Age'].mean()
        std = train_df.loc[(train_df['Title'] == title),'Age'].std()
        #cnt_nan = train_df.loc[(train_df['title'] == title),'Age'].isnull().sum()

        rand_1 = np.random.randint(avg - std, avg + std)
        

    except:
        #but if,we the cant recognise the title from training set, so it'll throw the error, we pass it to another method.
        #we we randoming the value around the mean of its sex (male/female).
        avg = train_df.loc[train_df['Sex'] == sex, 'Age'].mean()
        std = train_df.loc[train_df['Sex'] == sex, 'Age'].std()

        rand_1 = np.random.randint(avg - std, avg + std)
    
    return rand_1




#Action.
train_df.loc[train_df['Age'].isnull(),'Age'] = train_df[['Sex','Title']].loc[train_df['Age'].isnull()].apply(fillAge, axis = 1)
test_df.loc[test_df['Age'].isnull(),'Age'] = test_df[['Sex','Title']].loc[test_df['Age'].isnull()].apply(fillAge, axis = 1)

#so that's enough name column, we can drop it.
train_df.drop('Name', axis = 1, inplace = True)
test_df.drop('Name', axis = 1, inplace = True)




#Counting of each title.
train_title = train_df[['Title','PassengerId']].groupby('Title', as_index = False).count()
train_title.columns = ['Title','count_training']
#train_title




#Replacing Title into Closest Age Mean and Deviation (or maybe the nearest sounds like)

#mlle -> mrs
train_df['Title'] = train_df['Title'].replace('Mlle.','Mrs.')
test_df['Title'] = test_df['Title'].replace('Mlle.','Mrs.')

#mme -> mrs.
train_df['Title'] = train_df['Title'].replace('Mme.','Mrs.')
test_df['Title'] = test_df['Title'].replace('Mme.','Mrs.')

#ms. -> miss
train_df['Title'] = train_df['Title'].replace('Ms.','Miss.')
test_df['Title'] = test_df['Title'].replace('Ms.','Miss.')

#but we ignore uncommon title that maybe occour in test_df.




#Fill Embarked With Nearest value of Fare
train_df[train_df['Embarked'].isnull()]
#Logically, the nearest correlation between Embarked is the fare, so we want to know how the distribution of 'Embark' vs 'Fare'




fig, ax = plt.subplots(figsize = (15.7,6.27))
sns.barplot(x = 'Embarked', y = 'Fare', data = train_df)




#so the most logical thing, if the fare was 80 the nearest value and deviation is the 'C'
train_df['Embarked'] = train_df['Embarked'].fillna('C')




#filling the missing values in fare simply with mean of the 'S' values in Embarked in training datasets.
test_df[test_df['Fare'].isnull()]




test_df.loc[test_df['Fare'].isnull(), 'Fare'] = train_df.loc[train_df['Embarked'] == 'S','Fare'].mean()
test_df.iloc[152]




#We want to make title columns to more general. 
#So, we make most common Title are ['Mr.','Mrs.','Miss.','Master.', 'Dr.', 'Rev.'] and Cluster another uncommon title to one label -> 'Rare'.

import seaborn as sns

fig, ax = plt.subplots(figsize = (15.7,6.27))
sns.countplot(x = 'Title', data = train_df)




#make function to make new values.
nonRareTitle = ['Mr.','Mrs.','Miss.','Master.','Dr.','Rev.']
def generalizingTitle(title):
    if (title == nonRareTitle[0]) | (title == nonRareTitle[1]) | (title == nonRareTitle[2]) | (title == nonRareTitle[3]):
        title = title
    # and throw the uncommon title into 'Rare.'
    else:
        title = 'Rare.'
            
    return title




#results

train_df['Title'] = train_df['Title'].apply(generalizingTitle)
test_df['Title'] = test_df['Title'].apply(generalizingTitle)

#printing the results.
print(train_df['Title'].value_counts())
print('-'*10)
print(test_df['Title'].value_counts())




#we want to make new feature called 'Family', family represents the sum between columns 'Parch' and 'SibSp', 
#which's sum between #of parents and #of Siblings

train_df['Family'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['Family'] = test_df['SibSp'] + test_df['Parch'] + 1
#we add +1 bcs it counts with him/herself.




#we want to know the distribution number of Family
fig, ax = plt.subplots(figsize = (15.7,6.27))
sns.countplot(x = 'Family', data = train_df)

##of Family
grap = sns.FacetGrid(train_df, hue="Survived",aspect=4)
grap.map(sns.kdeplot,'Family',shade= True)
grap.set(xlim=(0, train_df['Family'].max()))
grap.add_legend()
#as we can see, the lower family number, if family number = 1. The passengger more likely not survived.




#so we classify the number of family.

def changeFamily(family):
    if family == 1:
        family = 'Alone'
    elif family >= 2 & family <= 5:
        family = 'Family'
    else:
        family = 'Huge Family'
    
    return family




train_df['Family'] = train_df['Family'].apply(changeFamily)
test_df['Family'] = test_df['Family'].apply(changeFamily)

#dropping unused columns
train_df.drop('SibSp', axis = 1, inplace = True)
train_df.drop('Parch', axis = 1, inplace = True)
#--
test_df.drop('SibSp', axis = 1, inplace = True)
test_df.drop('Parch', axis = 1, inplace = True)




# the person with lower age (child) more likely to survive then the person have higher number of age
# so we can create new value based on this rule.
# inserting new value child into sex columns.

grap = sns.FacetGrid(train_df, hue="Survived",aspect=4)
grap.map(sns.kdeplot,'Age',shade= True)
grap.set(xlim=(0, train_df['Age'].max()))
grap.add_legend()




#function to return child
def addChild(pasengger):
    sex,age = pasengger
    return 'child' if age < 16 else sex

#applying into columns sex,
train_df['Sex'] = train_df[['Sex','Age']].apply(addChild, axis = 1)
test_df['Sex'] = test_df[['Sex','Age']].apply(addChild, axis = 1)




#we drop passenger, because this column more likely to be a bug (i mean, make training accuracy very high if we keep this).
test_id = test_df.iloc[:,0]

train_df.drop('PassengerId', axis = 1, inplace = True)
test_df.drop('PassengerId', axis = 1, inplace = True)

#drop chaotic columns, cabin and ticket.
train_df.drop('Cabin', axis = 1, inplace = True)
test_df.drop('Cabin', axis = 1, inplace = True)

train_df.drop('Ticket', axis = 1, inplace = True)
test_df.drop('Ticket', axis = 1, inplace = True)




y_train = train_df.iloc[:,0]
x_train = train_df.iloc[:,1:]




columns = ['Sex','Embarked','Title','Family']
Y_columns = ['Survived']

#this function to get dummies columns from categorical value in dataframe in both train and test.
def exCategories(df):

    catg = df[columns]
    catg = pd.get_dummies(catg)
    
    nocatg = df.drop(columns, axis = 1)
    res = pd.concat([nocatg,catg], axis = 1)
    
    if ''.join(Y_columns) in res:
        y = res[Y_columns]
        x = res.drop(Y_columns, axis = 1)
    else:
        y = None
        x = res
        
    return y,x

y_train, x_train  = exCategories(train_df)
y_test, x_test = exCategories(test_df)




from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import cross_val_score




knn = KNeighborsClassifier()
logreg = LogisticRegression(random_state = 7, n_jobs= -1)
decisiontree = DecisionTreeClassifier(random_state = 7)
randomforest = RandomForestClassifier(random_state = 7, n_estimators= 100, n_jobs= -1)

classifiers = [knn,logreg,decisiontree,randomforest]




print('7-fold cross validation:\n')

for clf, label in zip(classifiers, 
                      ['KNN',
                       'Logistic Regression', 
                       'Decision Trees',
                       'Random Forest']):

    scores = cross_val_score(clf, x_train, y_train.Survived, cv=5, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" 
          % (scores.mean(), scores.std(), label))




from sklearn.grid_search import GridSearchCV

param_RF = {'max_depth' : [4, 5],
           'n_estimators': [50, 150, 100, 200, 300, 350, 400],
           'min_samples_split': [2, 3, 4],
           'min_samples_leaf': [2, 5]}

randomRF = GridSearchCV(randomforest, param_RF,cv = 5,n_jobs=-1)
randomRF.fit(x_train, y_train.Survived)




print(randomRF.best_params_)
print(randomRF.best_score_)




yres_train = randomRF.predict(x_train)




from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(accuracy_score(yres_train, y_train))
print(confusion_matrix(yres_train, y_train))




y_test = randomRF.predict(x_test)




res = pd.DataFrame(
    {'PassengerId': test_id,
     'Survived': y_test
    })
res.to_csv('res.csv', index = None)

