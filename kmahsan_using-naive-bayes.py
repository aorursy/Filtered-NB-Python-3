#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('reset', '')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#import the data files
url = "../input/train.csv"
train = pd.read_csv(url);
# ok now that the data set is loaded, lets take a peek at the dataset
#train_set = pd.get_dummies(train, columns=['Sex']);
print("\n"); print(train.sample(3).to_string()); print("\n");




fig = sns.factorplot(x="Sex", y="Survived", data=train,
                   size=4, aspect=3, kind="bar", palette="muted")
plt.title("The effect of the feature 'Sex' on survival")
plt.ylabel("Fraction Survived", FontSize=12)
plt.show()


ax = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                    size=4, aspect=3, kind="bar", palette="muted")
plt.title("The effect of the feature 'Pclass' on survival")
plt.ylabel("Fraction Survived", FontSize=12)
plt.show()

# check which rows contain NaN in Age column
boolColVec = train['Age'].isnull()
# now replace NaN values in 'Age' column with 0
train.loc[boolColVec, 'Age'] = np.double(0) 

# divide age into 4 categories
def age_class(x):
    if np.double(x)>= 0 and np.double(x) < 18:
        x = 1.0;
    elif np.double(x) >=18 and np.double(x)< 25:
        x = 2.0;
    elif np.double(x) >=25 and np.double(x) < 35:
        x = 3.0;
    else:
        x = 4.0;
    return x;
train.Age = train['Age'].apply(age_class)
fig = sns.factorplot(x="Age", y="Survived",data=train,
                    size=4,aspect=3, kind="bar", palette="muted")
plt.title("The effect of the feature 'Age' on survival")
plt.ylabel("Fraction Survived", FontSize=12)
plt.show()




# lets also drop the features we will not be using / keep only the features we will be using and see what the records look like
dummy = train[['PassengerId','Pclass', 'Sex', 'Age']];
print("\n"); print(dummy.head(2).to_string()); print("\n");




# pivot against 'Survived' feature, get a new data frame and get percent survived  
# To make the calculation easier we introduce a new column(feature) labeled  Unit which will have a value of 1 for each record (Passenger)
# When we use this 'Unit' feature with numpy sum() function we can easily see in the pivot table the total number of passengers belonging to 
# the categories Survived=0 and Survived=1
train['Unit'] = 1
# lets see what the dataset looks like after the introduction of the Unit column
print("\n"); print(train.sample(2).to_string()); print("\n");
print("=====================================================================================================================================================================")
pivot_survived  = pd.pivot_table(train,index=["Survived"], values=[ "Unit"],aggfunc=np.sum)
pivot_survived.reset_index(level=0, inplace=True) 
print("Total number of passengers belonging to each category under Survived features are shown in Unit column"); print("\n\n");
print(pivot_survived.to_string())




train.Sex = train['Sex'].map({'female':0,'male':1}).astype(int)




# pivot against 'Sex' feature, get a new data frame and get percent survived  
pivot_sex  = pd.pivot_table(train,index=["Survived", "Sex"], values=[ "Unit"],aggfunc=np.sum)
pivot_sex.reset_index(level=0, inplace=True) 
print("\n"); print(pivot_sex.to_string()); print("\n");




pivot_pclass = pd.pivot_table(train,index=["Survived", "Pclass"], values=[ "Unit"],aggfunc=np.sum)
pivot_pclass.reset_index(level=0, inplace=True) 
print("\n"); print(pivot_pclass.to_string()); print("\n");




pivot_Age = pd.pivot_table(train,index=["Survived", "Age"], values=[ "Unit"],aggfunc=np.sum)
pivot_Age.reset_index(level=0, inplace=True) 
print("\n"); print(pivot_Age.to_string()); print("\n"); 




get_ipython().run_cell_magic('latex', '', '\n\\begin{align}\nP(survived = 1\\ \\mathbf{GIVEN}\\ PassengerId=2) & =  \\\\\n\\Rightarrow P(survived=1) \\times P( sex = 0\\ |\\ survived = 1 )  \\times P( Pclass = 1\\ |\\ survived=1 ) \\times P ( Age = 4\\ |\\ survived = 1 )  & =  0.3841 \\times 0.6802 \\times 0.397 \\times 0.4249  =  0.0441\n\\end{align} \\\\\n\n\\begin{align}\nP(survived = 0\\ \\mathbf{GIVEN}\\ PassengerId=2) & = \\\\ \n\\Rightarrow P(survived=0) \\times P( sex = 0\\ |\\ survived = 0 ) \\times P( Pclass = 1\\ |\\ survived = 0 ) \\times P ( Age = 4\\ |\\ survived = 0 ) & = 0.6159 \\times 0.1488 \\times 0.147 \\times 0.4828 = 0.0065  \n\\end{align} ')




# load the training and test data sets
url1 = "../input/train.csv"
train = pd.read_csv(url1);
#lets check the data set 
print("\n"); print(train.drop(["Name"], axis=1).head(3).to_string()); print("\n");
url2 = "../input/test.csv"
test = pd.read_csv(url2);




def simple_transform(df1):
    # don't change the original input data set
    df = df1.copy()
    # drop records if sex is not defined or null 
    df = df.dropna(axis=0, how='any', subset=['Sex'])
    # map sex into binary values, male=1, female = 0
    df.Sex = df['Sex'].map({'female':0,'male':1}).astype(int)
    # check which rows for missing values in Age column
    boolColVec = pd.isnull(df['Age'])
    # now fill in the records with missing age values with age=0 
    df.loc[boolColVec, 'Age']= 0    
    # divide age into 4 categories
    def age_class(x):
        if np.double(x)>= 0 and np.double(x) < 18:
            x = 1.0;
        elif np.double(x) >=18 and np.double(x)< 25:
            x = 2.0;
        elif np.double(x) >=25 and np.double(x) < 35:
            x = 3.0;
        else:
            x = 4.0;
        return x;
    df.Age = df['Age'].apply(age_class)
    return df;




df_train = simple_transform(train)
df_test = simple_transform(test)




target_feature = df_train.loc[:, 'Survived'].as_matrix()




df_train = df_train[['PassengerId','Pclass', 'Sex', 'Age']];
# lets check everything is as we expect
print("First three records of the training data set"); print("\n");
print(df_train.head(3).to_string()); print("\n");
df_test = df_test[['PassengerId','Pclass', 'Sex', 'Age']];
print("First three records of the test data set");print("\n");
print(df_test.head(3).to_string()); print("\n");




train_mat = df_train.iloc[:, 1:4].as_matrix()
test_mat = df_test.iloc[:, 1:4].as_matrix()




from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(alpha=1)
mnb.fit(train_mat,target_feature)
print("\n"); print("The fraction of correctly predicted outcome from the test data set");
print(mnb.score(train_mat, target_feature)); print("\n");




print("\n"); print(" Multinomial Naive Bayes classifier prediction of the SUrvived feature for the first three records of training data set"); print("\n");
print(mnb.predict(train_mat[0:3])); print("\n");
Survived_test = mnb.predict(test_mat)




test['Survived'] = Survived_test
print(test.drop(['Name'], axis=1).sample(5).to_string())
print("\n")




my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test.Survived})
# submit the file
my_submission.to_csv('submission.csv', index=False)




train['Salutation'] = train['Name'].map(lambda dummy: dummy.split(',')[1].split('.')[0].strip()) 
print(train.drop(['Name'], axis=1).head(5).to_string())
print("\n All the Salutations that appeared in the names")
print(train['Salutation'].unique())
print("\n")




fig = sns.factorplot(x="Salutation", y="Survived", data=train,
                   size=3, aspect=6, kind="bar", palette="muted")
plt.title("The effect of the feature 'Salutation' on survival")
plt.ylabel("Fraction Survived", FontSize=12)
plt.show() 




# lets encode the Salutation into 7 numeric values
dict_sal = dict({'Mr':6, 'Mrs':2, 'Miss':3, 'Master':4, 'Don':7, 'Rev':7, 'Dr':5, 'Mme':1, 'Ms':1, 'Major':4, 'Lady':1, 'Sir':1, 'Mlle':1,'Col':5, 'Capt':7, 'the Countess':1, 'Jonkheer':7})

train['Salutation'] = train['Salutation'].map(dict_sal).astype(int)

fig = sns.factorplot(x="Salutation", y="Survived", data=train,
                   size=3, aspect=6, kind="bar", palette="muted")
plt.title("The effect of the feature 'Salutation' on survival")
plt.ylabel("Fraction Survived", FontSize=12)
plt.xlabel("Salutation encoded into integer values", FontSize=12)
plt.show() 




train.loc[train['Fare'].between(0, 30, inclusive=True), 'Fare'] = 0
train.loc[train['Fare'].between(30.0001, 513, inclusive=True), 'Fare'] = 1
print(train.drop(['Name'], axis=1).head(3).to_string())

