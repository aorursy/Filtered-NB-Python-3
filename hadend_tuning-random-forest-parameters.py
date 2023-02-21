#!/usr/bin/env python
# coding: utf-8



# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from time import time
import matplotlib.pyplot as plt
from operator import itemgetter

# Stop deprecation warnings from being printed
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




# Load training and test data into pandas dataframes
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# merge training and test sets into one dataframe
full = pd.concat([train, test])




# Get size of dataframes
for dataframe in [train, test, full]:
    print(dataframe.shape)




full.head(2)




#return a formatted percentage from a fraction
def percentage(numerator, denomenator):
    
    if type(numerator) == pd.core.series.Series:
        return (numerator/denomenator*100).map('{:.1f}%'.format)
    
    elif type(numerator) == int or type(numerator) == float:
        return '{:.1f}%'.format(float(numerator)/float(denomenator)*100) 
    
    else:
        print("check type")




#Get percentage by variable of values which are not NaN
percentage(full.count()-1, full.shape[0]-1)




# Get cabin #'s in list
cabin_numbers = full[full.Cabin.notnull()]['Cabin'].tolist()
cabin_numbers[:10]




# Number of passengers w/ cabin numbers by class
full[full.Cabin.notnull()].groupby('Pclass')['Pclass'].count()




# Percentage of passengers w/ cabin numbers by class
percentage(full[full.Cabin.notnull()].groupby('Pclass')['Pclass'].count(),
           full.groupby('Pclass')['Pclass'].count()
          )




# Number of passengers w/ cabin numbers by class AND survival
train[train.Cabin.notnull()].groupby(['Pclass', 'Survived'])['Cabin'].count()




# How classes were distributed by deck (decks were labeled A-G from top to bottom of ship)
full['Deck'] = full.Cabin.str.extract("([a-zA-Z])", expand=False)
full[full.Cabin.notnull()].groupby(['Deck', 'Pclass'])['Deck'].count()




# Remove 'Cabin' column from dataset
full = full.drop(['Cabin', 'Deck'], 1);




# find entries where port of embarkation is null
full[full.Embarked.isnull()]




# Extract first three characters from ticket number and look for partterns from port of embarkation
full['ticket_header'] = full.Ticket.str.extract("([a-zA-Z0-9]{3})", expand=False)
full.groupby(['ticket_header', 'Embarked'])['Ticket'].count().head(12)




full.groupby(['Sex', 'Pclass', 'Embarked'])['Embarked'].count().head(3)




full.set_value(61, 'Embarked', 'S');
full.set_value(829, 'Embarked', 'S');




full[full.Fare.isnull()]




# look at fare statistics for passengers traveling alone or with 1 spouse/sibling 
# and with ticket  beginning with '370'
full[full.ticket_header == '370'].groupby(['Parch', 'SibSp'])['Fare'].describe().head(8)




full.set_value(152, 'Fare', 7.75);




# remove 'ticket_header' and 'ticket' columns from data frame
full = full.drop(['ticket_header', 'Ticket'], 1);




# Extract title from name column using RE and put in new column labeled 'Title'
full['Title']= full.Name.str.extract("(.*, )([^\.]+)", expand=False)[1]
full.groupby('Title')['Name'].count()




# Convert French titles like Mlle and Mme to English equivalent and convert all other titles to 'Rare'
full.loc[full.Title == 'Mlle', 'Title'] = 'Miss'
full.loc[full.Title == 'Mme', 'Title'] = 'Mrs'
full.loc[~full.Title.isin(['Master', 'Mr', 'Mrs', 'Miss']), 'Title'] = 'Rare'




full.groupby('Title')['Name'].count()




# Look at survivorship numbers by title
full[full.Survived.notnull()].groupby(['Title', 'Survived'])['Name'].count()




# Look at median age by title
full.groupby(['Title'])['Age'].median()




# Look at standard deviation of age by title
full.groupby(['Title'])['Age'].std()




# Look at median age by title and class
full.groupby(['Title', 'Pclass'])['Age'].median()




full_ver2 = full.copy()




# create dataframe with median ages by class and title
age_summary = full.groupby(['Title', 'Pclass'])['Age'].median().to_frame()
age_summary = age_summary.reset_index()

for index in full_ver2[full_ver2.Age.isnull()].index:
    median = age_summary[(age_summary.Title == full_ver2.iloc[index]['Title'])&                          (age_summary.Pclass == full_ver2.iloc[index]['Pclass'])]['Age'].values[0]
    full_ver2.set_value(index, 'Age', median)




# format and split dataset for RF fitting.
def train_test_split(dataframe):
    try:
        # change gender labels to '0' or '1'
        dataframe["Sex"] = dataframe["Sex"].apply(lambda sex: 0 if sex == "male" else 1)
        
        convert = LabelEncoder()
        
        # change embarkation to numerical labels
        embarked = convert.fit_transform(dataframe.Embarked.tolist())
        dataframe['Embarked'] = embarked
        
        # change title to numerical labels
        title = convert.fit_transform(dataframe.Title.tolist())
        dataframe['Title'] = title    
        
    except:
        "dataframe is not correctly formatted"
    
    # split into training and test sets and move survival labels to list
    return dataframe[0:891].drop('Survived', 1),            dataframe[891:].drop('Survived', 1),            dataframe[0:891]['Survived'].tolist()




train_ver2, test_ver2, labels = train_test_split(full_ver2)




X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_ver2, labels, test_size=0.3, random_state=42);




# calculate the time to run a GridSearchCV for multiple numbers of parameter permutations.  
grid_times = {}
clf = RandomForestClassifier(random_state = 84)

features = X_train.columns.drop(['Name', 'PassengerId'], 1)

# I commented this out after running once locally since this block of code takes a long time to run
'''
for number in np.arange(2, 600, 50):
    
    param = np.arange(1,number,10)
    param_grid = {"n_estimators": param,
                  "criterion": ["gini", "entropy"]}
    
    grid_search = GridSearchCV(clf, param_grid = param_grid)
    
    t0 = time()
    grid_search.fit(X_train[features], y_train)
    compute_time = time() - t0
    grid_times[len(grid_search.grid_scores_)] = time() - t0
    
grid_times = pd.DataFrame.from_dict(grid_times, orient = 'index')
'''

# hard-coded values were found by running code above
grid_times = {0: { 2: 0.034411907196044922,
                  12: 1.5366179943084717,
                  22: 5.0431020259857178,
                  32: 11.378448963165283,
                  42: 20.211128950119019,
                  52: 30.040457010269165,
                  62: 39.442277908325195,
                  72: 56.834053993225098,
                  82: 67.847633838653564,
                  92: 91.005517959594727,
                  102: 111.2420859336853,
                  112: 135.75759792327881}}




final = pd.DataFrame.from_dict(grid_times)
final = final.sort_index()
plt.plot(final.index.values, final[0])
plt.xlabel('Number of Parameter Permutations')
plt.ylabel('Time (sec)')
plt.title('Time vs. Number of Parameter Permutations of GridSearchCV')




# function takes a RF parameter and a ranger and produces a plot and dataframe of CV scores for parameter values
def evaluate_param(parameter, num_range, index):
    grid_search = GridSearchCV(clf, param_grid = {parameter: num_range})
    grid_search.fit(X_train[features], y_train)
    
    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]
       
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
 
    plt.subplot(3,2,index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    return plot, df




# parameters and ranges to plot
param_grid = {"n_estimators": np.arange(2, 300, 2),
              "max_depth": np.arange(1, 28, 1),
              "min_samples_split": np.arange(1,150,1),
              "min_samples_leaf": np.arange(1,60,1),
              "max_leaf_nodes": np.arange(2,60,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}




index = 1
plt.figure(figsize=(16,12))
for parameter, param_range in dict.items(param_grid):   
    evaluate_param(parameter, param_range, index)
    index += 1




from operator import itemgetter

# Utility function to report best scores
def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")




# parameters for GridSearchCV
param_grid2 = {"n_estimators": [10, 18, 22],
              "max_depth": [3, 5],
              "min_samples_split": [15, 20],
              "min_samples_leaf": [5, 10, 20],
              "max_leaf_nodes": [20, 40],
              "min_weight_fraction_leaf": [0.1]}




grid_search = GridSearchCV(clf, param_grid=param_grid2)
grid_search.fit(X_train[features], y_train)

report(grid_search.grid_scores_, 4)




param_grid3 = {"n_estimators": [5, 40, 42],
              "max_depth": [5, 6],
              "min_samples_split": [5, 10],
              "min_samples_leaf": [3, 5],
              "max_leaf_nodes": [14, 15]}




grid_search = GridSearchCV(clf, param_grid=param_grid3)
grid_search.fit(X_train[features], y_train)

report(grid_search.grid_scores_, 4)




clf = RandomForestClassifier(min_samples_split = 40, 
                             max_leaf_nodes = 15, 
                             n_estimators = 40, 
                             max_depth = 5,
                             min_samples_leaf = 3)




clf.fit(train_ver2[features], labels);




prediction = clf.predict(test_ver2[features])

output = pd.DataFrame(test_ver2['PassengerId'])
output['Survived'] = prediction
output.to_csv('prediction.csv')

