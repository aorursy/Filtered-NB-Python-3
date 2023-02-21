#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




# data analysis and wrangling libraries
import pandas as pd
import numpy as np

#data visualization libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

#Scipy libraries (Statistical Analysis on data)
from scipy import stats
from statsmodels.formula.api import ols
from scipy.stats import zscore
import math

#Scikit Libraries (Machine Learning)

#preprocessing 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#model selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

#Models (Though a number of algorithms can be tested, we have used only few of them)
##Linear Models 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
##Neighbour Models
from sklearn.neighbors import KNeighborsClassifier
##Support Vector Machine Model
from sklearn.svm import SVC, LinearSVC
## Tree
from sklearn.tree import DecisionTreeClassifier
##Ensemble Models
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
##Neural Networ(sklearn)
from sklearn.neural_network import MLPClassifier
##XGBoost
import xgboost as xgb


#Metrics to evaluate the models
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, roc_curve, roc_auc_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('timeit', '')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)




get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\ntable.dataframe td, table.dataframe th {\n    border: 1px  black solid !important;\n  color: black !important;\n}')




train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')




train_df.head()




#creating a dummy dataset to explore without making any changes to original dataset
dataset = train_df.copy()




#Variable Distinction
display(dataset.info())
display(dataset.describe().transpose())




#Checking if our data is bias or not
plt.figure(figsize= (4,4))
sns.countplot(dataset['Survived'])




#selecting categorical categories
cat_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

# Ploting the count as well as percetage frequency
fig, axes = plt.subplots(3, 2, figsize = (12,14))
axes = axes.flatten()
    
for ax, cat in zip(axes, cat_features) :
    total = float(len(dataset[cat]))      
    sns.countplot(dataset[cat], palette = 'viridis', ax =ax)       
    for p in ax.patches :
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 10,
                '{:1.0f}%'.format((height/total) * 100), ha = 'center',)     
    plt.ylabel('Count')




#selecting numerical features 
num_features = ['Age', 'Fare']

# Ploting the distribution plot
fig, axes = plt.subplots(1, 2, figsize = (10,6))
axes = axes.flatten()

for ax, num in zip(axes, num_features) :
    sns.distplot(dataset[num], ax= ax, hist = True, kde = True)




corr = dataset.drop('PassengerId', axis =1).corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
display(sns.heatmap(data= corr, cmap= "coolwarm_r", robust= True, annot= True, square= True, mask= mask, fmt= "0.2f"))
display(train_df.corr()['Survived'].sort_values())




display(sns.countplot(dataset['Pclass'], hue = dataset['Survived'], palette= "viridis"))

ax = plt.gca()
# Iterate through the list of axes' patches
total = len(dataset['Pclass'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{:1.0f}'.format(height), 
            fontsize=12, ha='center', va='bottom')




dataset.corr()['Pclass'].sort_values()




fig = plt.figure(constrained_layout = True, figsize = (10,5))
#Creating a grid of 2 cols and 1 rows
grid = gridspec.GridSpec(ncols= 2, nrows=1, figure= fig)

#Plot Pclass vs Fare
ax1 = fig.add_subplot(grid[0,0])
ax1.set_title('Pclass vs Fare')
sns.barplot(x="Pclass",y="Fare",data=dataset, palette = "viridis", ax= ax1)

#Plot Pclass vs Age
ax1 = fig.add_subplot(grid[0,1])
ax1.set_title('Pclass vs Age')
sns.barplot(x="Pclass",y="Age",data=dataset, palette = "viridis", ax= ax1)




fig = plt.figure(constrained_layout = True, figsize = (10,5))
#Creating a grid of 2 cols and 1 rows
grid = gridspec.GridSpec(ncols= 2, nrows=1, figure= fig)

#Distribution plot 
ax1 = fig.add_subplot(grid[0,0])
ax1.set_title('Distribution')
sns.distplot(dataset["Age"], label="Skewness : %.1f"%(dataset["Age"].skew()), ax= ax1)

#Box plot of Age with Survival 
ax1 = fig.add_subplot(grid[0,1])
ax1.set_title('Pclass vs Age')
sns.boxplot(x="Survived", y="Age",data=dataset, palette = "viridis", ax= ax1)




dataset[dataset['Survived'] == 1]['Age'].hist(alpha = 0.6, bins = 10, color = 'g')
dataset[dataset['Survived'] == 0]['Age'].hist(alpha = 0.5, bins = 10, color = 'b')




#let us try to split the age into categories
dataset['Age_Category'] = pd.cut(dataset['Age'], bins= 8,labels = ['0-10', '10-20', '20-30', '30-40', '40-50', 
                                                                    '50-60','60-70', '80-90'])
plt.figure(figsize= (12,6))
sns.countplot(dataset['Age_Category'], hue= dataset['Survived'])




fig = plt.figure(constrained_layout = True, figsize = (10,5))
#Creating a grid of 2 cols and 1 rows
grid = gridspec.GridSpec(ncols= 2, nrows=1, figure= fig)

#Box plot of Sex with Survived
ax1 = fig.add_subplot(grid[0,0])
ax1.set_title('Sex vs Survived')
sns.countplot(dataset['Sex'], hue = dataset['Survived'], palette= "viridis", ax= ax1)

#Box plot of Age with Survival 
ax1 = fig.add_subplot(grid[0,1])
ax1.set_title('Sex vs Pclass')
sns.countplot(dataset['Sex'], hue = dataset['Pclass'], palette= "viridis", ax= ax1)




#selecting categorical categories
family_features = ['SibSp', 'Parch']

# Ploting the count as well as percetage frequency
fig, axes = plt.subplots(1, 2, figsize = (14,6))
axes = axes.flatten()
    
for ax, cat in zip(axes, family_features) :
    total = float(len(dataset[cat]))      
    sns.countplot(dataset[cat], palette = 'viridis', ax =ax, hue= dataset['Survived'])
    for p in ax.patches :
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 10,
                '{:1.0f}'.format(height), ha = 'center',)     
    plt.ylabel('Count')




dataset['Family'] = dataset.apply(lambda x : x.SibSp + x.Parch, axis = 1)

sns.countplot(dataset['Family'], hue = dataset['Survived'], palette= "viridis")
ax = plt.gca()

# Iterate through the list of axes' patches
total = len(dataset['Pclass'])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{:1.0f}'.format(height), 
            fontsize=12, ha='center', va='bottom')




fig = plt.figure(constrained_layout = True, figsize= (10,10))
grid = gridspec.GridSpec(ncols= 2, nrows= 2, figure= fig)

#Count plot of Embarked with Survival 
ax1 = fig.add_subplot(grid[0,0])
ax1.set_title('Embarked with Survival')
sns.countplot(x=dataset["Embarked"], hue= dataset["Survived"], palette = "viridis", ax= ax1)

#Box plot of Embarked with Age 
ax1 = fig.add_subplot(grid[0,1])
ax1.set_title('Embarked with Age')
sns.boxplot(x="Embarked", y="Age", data=dataset, palette = "viridis", ax= ax1)

#Box plot of Embarked with Fare 
ax1 = fig.add_subplot(grid[1,0])
ax1.set_title('Embarked with Fare')
sns.boxplot(x="Embarked", y="Fare", data=dataset, palette = "viridis", ax= ax1)

#Box plot of Embarked with Pclass 
ax1 = fig.add_subplot(grid[1,1])
ax1.set_title('Embarked with Pclass')
sns.countplot(dataset["Embarked"], hue =dataset['Pclass'], palette = "viridis", ax= ax1)




fig = plt.figure(constrained_layout = True, figsize= (12,6))
grid = gridspec.GridSpec(ncols= 2, nrows= 1, figure= fig)

#Distribution of Fare
ax1 = fig.add_subplot(grid[0,0])
ax1.set_title('Distribution of Fare')
sns.distplot(dataset["Fare"], label="Skewness : %.1f"%(dataset["Fare"].skew()))

#Box plot of Fare with Survival 
ax1 = fig.add_subplot(grid[0,1])
ax1.set_title('Fare with Survival')
sns.boxplot(x="Survived", y="Fare", data=dataset, palette = "viridis", ax= ax1)




dataset[dataset['Pclass']== 1]['Fare'].hist(bins = 20)
dataset[dataset['Pclass']== 2]['Fare'].hist(bins = 20)
dataset[dataset['Pclass']== 3]['Fare'].hist(bins = 20)




dataset['Cabin'].fillna('None', inplace = True)
dataset['Have Cabin'] = dataset['Cabin'].apply(lambda x : "Yes" if x != 'None' else "No")




sns.countplot(dataset['Have Cabin'], hue = dataset['Survived'])




merged_df = pd.concat([train_df, test_df], sort= False).reset_index(drop= True)
merged_df.head(3)




merged_df.info()
display(sns.heatmap(merged_df.isna(), cmap= 'binary_r', yticklabels = False))




# get mean, standard deviation and number of null values
mean = train_df["Age"].mean()
std = train_df["Age"].std()

def impute_age(age) :   
    if age != age :
        return np.random.randint(mean - std, mean + std)
    else :
        return age
    
merged_df['Age'] = merged_df['Age'].apply(lambda x : impute_age(x))




#identify the missing Fare value passenger
display(merged_df[merged_df['Fare'].isna()])

#identify the missing Embarked value passenger
display(merged_df[merged_df['Embarked'].isna()])

#identify aggregate value of Fare under depending section
display(merged_df.groupby(['Pclass', 'Embarked']).mean()['Fare'])

#fill the relevant value of the Fare for the Passenger
merged_df.loc[1043, 'Fare'] = 14

#fill the relevant value of the Embarked for the Passenger
merged_df.loc[61, 'Embarked'] = 'S'
merged_df.loc[829, 'Embarked'] = 'S'




#drop the extra columns
merged_df.drop('PassengerId', axis= 1, inplace= True)




merged_df['Name'].head(5)




merged_df['Title'] = merged_df['Name'].apply(lambda x : x.split(',')[1].split(' ')[1].strip())
display(merged_df['Title'].unique())
display(sns.countplot(merged_df['Title']))




merged_df["Title"] = merged_df["Title"].replace(['Don.', 'Rev.', 'Dr.', 'Mme.','Ms.', 'Major.', 'Lady.', 
                                                 'Sir.', 'Mlle.', 'Col.', 'Capt.', 'the','Jonkheer.', 'Dona.'], 'Other')
display(sns.countplot(merged_df['Title']))




sns.factorplot(x="Title",y="Survived",data=merged_df,kind="bar", order = ['Mr.', 'Other', 'Master.', 'Miss.', 'Mrs.'])




#convert them into ordinal values
merged_df['Title'] = merged_df['Title'].map({'Mr.' : 0, 'Other' : 1,'Master.' : 2,'Miss.' : 3,'Mrs.' : 4})




#name length
merged_df['Name Length'] = merged_df['Name'].apply(lambda x : len(x))
display(sns.factorplot(x ='Survived', y= 'Name Length',kind = 'bar', data = merged_df))




#drop name variable 
merged_df.drop('Name', axis= 1, inplace= True)




#creating family feature
merged_df['Family'] = merged_df.apply(lambda x : x.SibSp + x.Parch, axis=1)

#creating into categorical feature
merged_df['Family Size'] = merged_df['Family'].apply(lambda x : "Alone" if x == 0 else "Small" if x < 4 else "Large")

display(sns.factorplot(y ='Survived', x= 'Family Size',kind = 'bar', data = merged_df))




#converting family size to categorical features
fam_features = pd.get_dummies(merged_df['Family Size'], drop_first= True, prefix= "Fam")
merged_df = pd.concat([merged_df, fam_features], axis= 1)

#drop Parch, Sibsp & Family feature
merged_df.drop(['Parch', 'SibSp', 'Family', 'Family Size'], axis= 1, inplace = True)




#imputing NA values with 0
merged_df['Cabin'].fillna(0, inplace = True)

#changing other values to 1
merged_df['Cabin'] = merged_df['Cabin'].apply(lambda x : 0 if x ==0 else 1)

display(sns.countplot(merged_df['Cabin'], hue = merged_df['Survived']))




merged_df['Ticket'].head()




def ticket_prefix(ticket) :
    #if ticket is alphabetic(not numeric)
    if not ticket.isdigit() :
        return ticket.replace(".","").replace("/","").strip().split(' ')[0]
    else :
        return 'No'

merged_df['Ticket prefix int'] = merged_df['Ticket'].apply(lambda x : x.split()[0])
merged_df['Ticket prefix'] = merged_df['Ticket prefix int'].apply(lambda x : ticket_prefix(x))




display(merged_df['Ticket prefix'].nunique())
display(merged_df['Ticket prefix'].unique())




#converting to categorical feature
ticket_feature = pd.get_dummies(merged_df['Ticket prefix'], drop_first= True, prefix= "Tic")

#merging the tables
merged_df = pd.concat([merged_df, ticket_feature], axis=1)

#drop Ticket columns
merged_df.drop(['Ticket', 'Ticket prefix int', 'Ticket prefix'], axis =1 ,inplace= True)




# Create categorical values for Pclass
merged_df["Pclass"] = merged_df["Pclass"].astype("category")
pclass_feature = pd.get_dummies(merged_df["Pclass"],prefix="Pc", drop_first= True)
merged_df = pd.concat([merged_df, pclass_feature], axis= 1)

# Create categorical values for Sex
sex_feature = pd.get_dummies(merged_df["Sex"],prefix="Sex", drop_first= True)
merged_df = pd.concat([merged_df, sex_feature], axis= 1)

# Create categorical values for Embarked
embarked_feature = pd.get_dummies(merged_df["Embarked"],prefix="Em", drop_first= True)
merged_df = pd.concat([merged_df, embarked_feature], axis= 1)

#drop the duplicate columns
merged_df.drop(['Pclass', 'Sex', 'Embarked'],axis= 1, inplace= True)




merged_df['Age_Category'] = pd.cut(merged_df['Age'], bins= 8,labels = ['0-10', '10-20', '20-30', '30-40', '40-50', 
                                                                    '50-60','60-70', '80-90'])

features = pd.get_dummies(merged_df['Age_Category'], drop_first= True, prefix= 'Age')

merged_df = pd.concat([merged_df, features], axis=1)

merged_df.drop(['Age', 'Age_Category'], axis=1, inplace= True)




#spliting back the data into training and test data
training_data = merged_df.iloc[:891,:]  
test_data = merged_df.iloc[891 :,:] 
test_data.drop('Survived', axis =1, inplace =True)

X_train = training_data.drop('Survived', axis = 1).values
y_train = training_data['Survived'].values

X_test = test_data.values

display(X_train.shape)
display(y_train.shape)
display(X_test.shape)




#create a scaling object
scaler = MinMaxScaler()

#fit the object only on training data and not test data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)




#write a function to evaluate your model

def evaluate_accuracy(model) :
    
    print(str(model))
    #On Training Data
    model.fit(X_train, y_train)
    predict_train = model.predict(X_train)
    training_accuracy = accuracy_score(y_train, predict_train)
    print("Training Data")
    print(confusion_matrix(y_train, predict_train))
    print(f'Accuracy Score: {training_accuracy}')
    print('-'*50)
    print("Validation Data")
    predict_score = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy')
    validation_accuracy = predict_score.mean()
    print(f'Accuracy Score: {validation_accuracy}')
    print('')
    return training_accuracy, validation_accuracy




models = []
k = 42
#Linear Class
logistic_clf = LogisticRegression(random_state= k)
ridge_clf = RidgeClassifier(random_state= k)
models.append(logistic_clf)
models.append(ridge_clf)

#Neighbor Class
knn_clf = KNeighborsClassifier()
models.append(knn_clf)

#SVC Class
svc_clf = SVC(random_state= k)
linearsvc_clf = LinearSVC(random_state= k)
models.append(svc_clf)
models.append(linearsvc_clf)

#Tree Class
tree_clf = DecisionTreeClassifier(random_state= k)
models.append(tree_clf)

#Ensemble
randomforest_clf = RandomForestClassifier(random_state= k)
bagging_clf = BaggingClassifier(random_state= k)
gradboosting_clf = GradientBoostingClassifier(random_state= k)
adaboosting_clf = AdaBoostClassifier(DecisionTreeClassifier(random_state= k),random_state= k)
models.append(randomforest_clf)
models.append(bagging_clf)
models.append(gradboosting_clf)
models.append(adaboosting_clf)

#Neural Network
mlp_clf = MLPClassifier(random_state = k)
models.append(mlp_clf)

#Xgboost
xgboost = xgb.XGBClassifier(random_state = k)
models.append(xgboost)




training_accuracy = []
validation_accuracy = []

for model in models :
    train_acc, val_acc = evaluate_accuracy(model)
    training_accuracy.append(train_acc)
    validation_accuracy.append(val_acc)




result = pd.DataFrame({'Algorithm' : ['Logistic', 'Ridge', 'KNN', 'SVC', 'Lin SVC', 'Tree', 
                                     'Rnd Forest', 'Bagging', 'Grad Boost', 'AdaBoost', 'MLP', 'XGBoost'], 
                       'Training Accuracy': training_accuracy, 'Validation Accuracy' : validation_accuracy})

display(result)




sns.barplot(x = 'Algorithm',  y = 'Validation Accuracy', data = result)
ax = plt.gca()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{:1.3f}'.format(height), 
            fontsize=12, ha='center', va='bottom')




sns.scatterplot(x = 'Training Accuracy', y='Validation Accuracy', data = result)




#tuning the model

svc_classifier = SVC()
param_grid = {'C': [0.1, 1, 3], 'kernel': ['rbf', 'linear'], 'gamma' : ['scale', 0.1, 0.01], 
              'degree' : [1 ,3,5], 'break_ties': ['True', 'False']}

svc_grid = GridSearchCV(svc_classifier, param_grid, scoring= "accuracy", cv= 10)
svc_grid.fit(X_train, y_train)
print(f'Best Score: {svc_grid.best_score_:0.5f}')
print(f'Best Parameter: {svc_grid.best_params_}')




optimized_svc = SVC(C=1.0, break_ties=True, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=1, gamma='scale', kernel='rbf',
                    max_iter=-1, probability=True, random_state=42, shrinking=True, tol=0.001,verbose=False)

scores = cross_val_score(optimized_svc, X_train, y_train, cv = 10)
print(f'Tuned SVC Score: {scores.mean()}')




# Gradient boosting tunning

gb_clf= GradientBoostingClassifier(random_state= 42)

#param_grid = {'loss' : ["deviance"], 'n_estimators' : [100,300, 500, 750], 'learning_rate': [0.3 ,0.1],
#              'max_depth': [8,10,12], 'min_samples_leaf': [50,75,100],'max_features': [0.3, 0.1]  }

''' Best Parameters from first grid are : {'learning_rate': 0.3, 'loss': 'deviance', 'max_depth': 10,
 'max_features': 0.3, 'min_samples_leaf': 100, 'n_estimators': 500}
'''

param_grid_2 = {'loss' : ["deviance"], 'n_estimators' : [500, 600], 'learning_rate': [0.3, 0.5, 0.7],
               'max_depth': [9,10,11], 'min_samples_leaf': [100, 150],'max_features': [0.3, 0.5, 0.7]}

param_grid_3 = {'loss' : ["deviance"], 'n_estimators' : [600], 'learning_rate': [0.3, 0.4],
               'max_depth': [9], 'min_samples_leaf': [100, 120],'max_features': [0.3, 0.4]}

gb_grid = GridSearchCV(gb_clf, param_grid_3, cv=10, scoring="accuracy", n_jobs=-1)
gb_grid.fit(X_train,y_train)
print(f'Best Score: {gb_grid.best_score_:0.5f}')
print(f'Best Parameter: {gb_grid.best_params_}')




optimized_gb = GradientBoostingClassifier(learning_rate =  0.3, max_depth = 9, max_features = 0.4, min_samples_leaf =  100, 
                                          n_estimators = 700, random_state= 42)

scores = cross_val_score(optimized_gb, X_train, y_train, cv = 10)
print(f'Tuned Gradient Boosting Score: {scores.mean()}')




# RFC Parameters tunning 
rf_clf = RandomForestClassifier()


## Search grid for optimal parameters
param_grid = {"max_depth": [None],"max_features": [1,10, 15, 18], "min_samples_split": [5, 10, 12],
              "min_samples_leaf": [1, 3],"bootstrap": [True, False],"n_estimators" :[300, 750],
              "criterion": ["gini"]}


rf_grid = GridSearchCV(rf_clf,param_grid = param_grid, cv=10, scoring="accuracy")

rf_grid.fit(X_train,y_train)
rf_grid.best_score_




optimized_rf = RandomForestClassifier(bootstrap = False, criterion = 'gini', max_depth = None, max_features = 15, 
                                      min_samples_leaf = 1, min_samples_split = 10, n_estimators = 750, random_state=42)

scores = cross_val_score(optimized_rf, X_train, y_train, cv = 10)
print(f'Tuned Random Forest Score: {scores.mean()}')




#creating training and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)




from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping




X_train.shape




model = Sequential()

# Adding the input layer and first hidden layer
model.add(Dense(input_dim=54, units=32, activation='relu'))
model.add(Dropout(rate=0.75)) 

# Adding the second hidden layer
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(rate=0.5))

# Adding the output layer
model.add(Dense(units=1, activation='sigmoid'))

#Early Stopping
early_stop  = EarlyStopping(monitor= 'val_loss', mode = 'min', patience= 25, verbose=1)

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train, y_train, batch_size=10, epochs=500, validation_data = (X_val,y_val), verbose = 3,
          callbacks = [early_stop])




history = pd.DataFrame(model.history.history)

fig, axes = plt.subplots(2,1, figsize = (10,8))

axes[0].plot(history['loss'], 'r', label = 'Training Loss')
axes[0].plot(history['val_loss'], 'b', label = 'Validation Loss')
legend  = axes[0].legend()

axes[1].plot(history['accuracy'], 'r', label = 'Training Accuracy')
axes[1].plot(history['val_accuracy'], 'b', label = 'Validation Accuracy')
legend  = axes[1].legend()




#fit the model on complete data
random_forest = RandomForestClassifier(bootstrap = False, criterion = 'gini', max_depth = None, max_features = 15, 
                                      min_samples_leaf = 1, min_samples_split = 10, n_estimators = 750, random_state=42)

random_forest.fit(X_train, y_train)

#predict for the test set
predictions = random_forest.predict(X_test)




submission = pd.DataFrame(columns= ['PassengerId', 'Survived'])
submission['PassengerId'] = test_df['PassengerId']
submission['Survived'] = predictions
submission['Survived'] = submission['Survived'].astype('int64')




submission.to_csv("Optimized Random Forest.csv",index=False)




predictions = model.predict_classes(X_test)




submission = pd.DataFrame(columns= ['PassengerId', 'Survived'])
submission['PassengerId'] = test_df['PassengerId']
submission['Survived'] = predictions
submission['Survived'] = submission['Survived'].astype('int64')




submission.to_csv("Neural Network.csv",index=False)






