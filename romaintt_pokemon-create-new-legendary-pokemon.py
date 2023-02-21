#!/usr/bin/env python
# coding: utf-8



# Let's check our input file
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. I like it most for plot
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.cross_validation import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model




# Import data into a Pandas Dataframe
data = pd.read_csv("../input/pokemon.csv")




# Let's check the columns we have
data.info()




# Let's see how data look like
data.head()




data['type2'].fillna('None', inplace=True)




# Replace strings by NaN value, and convert numbers into an actual number type
data['capture_rate'] = pd.to_numeric(data['capture_rate'], errors='coerce')




# Create the type <-> ID conversion dictionary
all_types = set(data['type1']).union(set(data['type2']))
type_id_dict = dict(zip(list(all_types), range(len(all_types))))

all_abilities = set(data['abilities'])
abilities_id_dict = dict(zip(all_abilities, range(len(all_abilities))))

all_classf = set(data['classfication'])
classf_id_dict = dict(zip(all_classf, range(len(all_classf))))




# Convert types in the Dataframe
for type_name, type_id in type_id_dict.items():
    data['type1'].replace(type_name, type_id, inplace=True)
    data['type2'].replace(type_name, type_id, inplace=True)
    
for ability_name, ability_id in abilities_id_dict.items():
    data['abilities'].replace(ability_name, ability_id, inplace=True)

for classf_name, classf_id in classf_id_dict.items():
    data['classfication'].replace(classf_name, classf_id, inplace=True)
   
data[['type1', 'type2', 'classfication', 'abilities']].head()




in_combat_col = ['against_bug', 'against_dark', 'against_dragon',
       'against_electric', 'against_fairy', 'against_fight', 'against_fire',
       'against_flying', 'against_ghost', 'against_grass', 'against_ground',
       'against_ice', 'against_normal', 'against_poison', 'against_psychic',
       'against_rock', 'against_steel', 'against_water', 'attack',
       'base_total', 'defense', 'hp', 'sp_attack', 'sp_defense', 'speed', 
        'type1', 'type2']
off_combat_col = ['abilities', 'base_egg_steps', 'base_happiness', 'capture_rate',
       'classfication', 'experience_growth', 'height_m', 'percentage_male', 
       'weight_kg', 'is_legendary']
out_of_gameplay_col = ['japanese_name', 'name', 'pokedex_number', 'generation']

# Check that the list do not overlap amongst themselves
if not set(in_combat_col).intersection(set(off_combat_col))    and not set(in_combat_col).intersection(set(out_of_gameplay_col))    and not set(off_combat_col).intersection(set(out_of_gameplay_col)):
    print("☑ Lists do not overlap :)")
else:
    print("Lists overlap !")




# Let's find redondant information with a simple cross-correlation
corr = data[in_combat_col].corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f',annot_kws={'size': 10},
            xticklabels=in_combat_col, yticklabels=in_combat_col, cmap= 'coolwarm')




# Remove base_total
data.drop("base_total", axis=1, inplace=True)
in_combat_col.remove("base_total")




# Let's do the same with off-combat parameters
corr = data[off_combat_col].dropna(axis=0).corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr, cbar=True, square=True, annot=True, fmt='.2f',annot_kws={'size': 15},
            xticklabels=off_combat_col, yticklabels=off_combat_col, cmap= 'coolwarm')




# Let's see the number of legendary pokemon in data
sns.countplot(data['is_legendary'])




# Let's find relations between beeing a legendary pokemon or not, and the basic combat parameters
# We do not use in_combat_col because there are too many columns to plot
basic_in_combat_col = ['attack', 'defense', 'hp', 'sp_attack', 'sp_defense', 'speed']

sns.pairplot(data, vars=basic_in_combat_col, hue="is_legendary")




# Let's compute the sum of these parameters and store them in a new column
# We first need to normalize the data to do a relevant sum
basic_combat_df = data[basic_in_combat_col]
norm_data = (basic_combat_df - basic_combat_df.mean()) / (basic_combat_df.max() - basic_combat_df.min())
norm_data['sum'] = norm_data.sum(axis=1, numeric_only=True)
# Add the is_legendary column
norm_data['is_legendary'] = data['is_legendary']
norm_data.head()




# Let's display the repartition of legendary pokemon in funciton of this sum
sns.boxplot(data=norm_data, x='is_legendary', y='sum')




other_in_combat_col = ['against_psychic', 'against_grass', 'against_flying', 
                       'against_ground', 'against_water', 'against_electric', 
                       'against_fire', 'against_fairy', 'against_dark', 'against_ice', 
                       'against_steel', 'against_bug', 'against_normal', 'against_poison', 
                       'against_ghost', 'against_rock', 'against_fight', 'against_dragon']




# Let's compute the sum of these parameters and store them in a new column
# We first need to normalize the data to do a relevant sum
other_in_combat_df = data[other_in_combat_col]
norm_data = (other_in_combat_df - other_in_combat_df.mean()) / (other_in_combat_df.max() - other_in_combat_df.min())
norm_data['sum'] = norm_data.sum(axis=1, numeric_only=True)
# Add the is_legendary column
norm_data['is_legendary'] = data['is_legendary']
norm_data.head()




# Let's display the repartition of legendary pokemon in funciton of this sum
sns.boxplot(data=norm_data, x='is_legendary', y='sum')




# Let's display the repartition of legendary pokemon in function of types
fig, axs = plt.subplots(ncols=2, figsize=(12,4))
sns.boxplot(ax=axs[0], data=data[['type1','is_legendary']], x='is_legendary', y='type1')
sns.boxplot(ax=axs[1], data=data[['type2','is_legendary']], x='is_legendary', y='type2')




fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(16,16))
for i, col in enumerate([c for c in off_combat_col if c !="is_legendary"]):
    sns.swarmplot(ax=axs[i//3][i%3], data=data, x='is_legendary', y=col)




d = {}
d['with_everything'] = data['is_legendary']
d['without_nan_capture'] = data.dropna(axis=0, how='any', subset=['capture_rate'])['is_legendary']
d['without_nan_weight'] = data.dropna(axis=0, how='any', subset=['weight_kg'])['is_legendary']
d['without_nan_height'] = data.dropna(axis=0, how='any', subset=['height_m'])['is_legendary']
d['without_nan_height_weight'] = data.dropna(axis=0, how='any', subset=['height_m', 'weight_kg'])['is_legendary']
d['without_nan_male'] = data.dropna(axis=0, how='any', subset=['percentage_male'])['is_legendary']
for key, s in d.items():
    print("{} : \n{}".format(key, s.value_counts()))

legendary_serie_without_nan = data.dropna(axis=0, how='any')['is_legendary']
legendary_serie_with_nan = data['is_legendary']




useful_off_combat_col = off_combat_col.copy()
for col in ['abilities', 'classfication', 'is_legendary', 'percentage_male']:
    if col in useful_off_combat_col:
        useful_off_combat_col.remove(col)

# Drop rows where there is a NaN in height_m, weight_kg, or capture_rate
subset = ['height_m', 'weight_kg', 'capture_rate']
data = data.dropna(axis=0, how='any', subset=subset)




#now split our data into train and test
train, test = train_test_split(data, test_size=0.3)# in this our main data is splitted into train and test
# we can check their dimension
print(train.shape)
print(test.shape)

train_y = train['is_legendary']# This is output of our training data
test_y = test['is_legendary']   #output value of test dat




prediction_var = basic_in_combat_col.copy()




train_X = train[prediction_var]# taking the training data input 
test_X= test[prediction_var] # taking test data inputs




model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train_X,train_y)# now fit our model for training data
prediction=model.predict(test_X)# predict for the test data
metrics.accuracy_score(prediction,test_y) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values




model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)




prediction_var = useful_off_combat_col.copy()
prediction_var




train_X = train[prediction_var]# taking the training data input 
test_X = test[prediction_var] # taking test data inputs




np.isfinite(train_X).all()




model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train_X,train_y)# now fit our model for training data
prediction=model.predict(test_X)# predict for the test data
metrics.accuracy_score(prediction,test_y) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values




model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)




prediction_var = (useful_off_combat_col + basic_in_combat_col).copy()
prediction_var




train_X = train[prediction_var]# taking the training data input 
test_X = test[prediction_var] # taking test data inputs




model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train_X,train_y)# now fit our model for training data
prediction=model.predict(test_X)# predict for the test data
metrics.accuracy_score(prediction,test_y) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values




sorted_features = sorted(zip(model.feature_importances_, train_X.columns), reverse=True)
unzip_sorted_features = list(zip(*sorted_features))
labels = unzip_sorted_features[1]
scores = unzip_sorted_features[0]

fig1, ax1 = plt.subplots()
fig1.suptitle("Importance of features in Random Forest algorithm")
fig1.set_figheight(10)
fig1.set_figwidth(10)
ax1.pie(scores, labels=labels, autopct='%1.2f%%',
        shadow=False, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()




model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train_X[['base_egg_steps', 'capture_rate', 'base_happiness']],
          train_y)# now fit our model for training data
prediction=model.predict(test_X[['base_egg_steps', 'capture_rate', 'base_happiness']])# predict for the test data
metrics.accuracy_score(prediction,test_y) # to check the accuracy






