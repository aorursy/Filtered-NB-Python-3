#!/usr/bin/env python
# coding: utf-8



# Make sure to have the library versions below for interactive plotting
from IPython.display import clear_output
get_ipython().system('pip install cufflinks')
clear_output() # Clears out huge shell output from cufflinks installation...




# Data Wrangling
import os
import re
import sys
import itertools
import numpy as np
import pandas as pd
import random as rnd

# Visualizations - Regular plotting
import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,6)
get_ipython().run_line_magic('matplotlib', 'inline')

# Visualizations - Interactive plotting
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=False)
import cufflinks as cf
cf.go_offline()

# Feature Engineering
import scipy.stats as st
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Model Evaluation
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Warning Handling
import warnings
warnings.filterwarnings(action='ignore')

print(f'Python environment: {sys.version}')




# Load from .csv
train_df = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')

# Extract survival information from train_df and combine the  rest
has_survided = train_df['Survived']
train_df.drop('Survived', axis=1, inplace=True)
df = pd.concat([test_df, train_df])

# Let's not forget to save train and test index for splitting it later
train_index, test_index = train_df.index, test_df.index 

# As we no longer need these dataframes, let's clear some memory
del train_df, test_df

# And them let's have a look at what we've got
df.head()




df.info()




# Let's create some features that might be insightful and easier to grasp during our exploratory analysis
df['FamilySize'] = 1 + df.SibSp + df.Parch
df['NameLength'] = df.Name.apply(len)
df['TravelsAlone'] = df.FamilySize.apply(lambda x: 1 if x == 1 else 0)

# I noticed every name has a prefixed title, maybe that's predictive in some way... Let's extract it and find it out!
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
df.Title.value_counts(dropna=False).iplot('bar', title='Captured Titles from Names')




title_dict = {
    'Mrs': 'Mrs', 'Lady': 'Mrs', 'Countess': 'Mrs',
    'Jonkheer': 'Other', 'Col': 'Other', 'Rev': 'Other',
    'Miss': 'Miss', 'Mlle': 'Miss', 'Mme': 'Miss', 'Ms': 'Miss', 'Dona': 'Miss',
    'Mr': 'Mr', 'Dr': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Master': 'Mr'
}

df.Title = df.Title.map(title_dict)
print('Title count')
df.Title.value_counts(dropna=False)




# Maybe de distribuition of age might help us
df[['Title', 'Age']].pivot(columns='Title', values='Age').iplot(kind='box', title='Age Distribution across Titles')




# What about Fares and Pclass
df[['Pclass', 'Fare']].pivot(columns='Pclass', values='Fare').iplot(kind='box', title='Fare Distribution across Ticket Classes')




# Average Age per Title is the best we can get in a quick-fix
for title in df.Title.unique():
    df.loc[(df.Age.isnull())&(df.Title==title), 'Age'] = df.Age[df.Title==title].mean()

# Average Fare per Pclass as well... The boxplots were indeed insightful
for pclass in df.Pclass.unique():
    df.loc[(df.Fare.isnull())&(df.Pclass==pclass), 'Fare'] = df.Fare[df.Pclass==pclass].mean()

# Let's just take the most common value (mode) to fill Embarked... It was just one missing piece after all
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])




# Now, let's turn our categorical features into numerical ones so we can plug them into our Machine Learning Models
df['Sex'], int2class_sex           = pd.factorize(df.Sex)
df['Title'], int2class_title       = pd.factorize(df.Title)
df['Embarked'], int2class_embarked = pd.factorize(df.Embarked)

# To easily navigate from numeric to categorical variables, let's re-write those lists as dictionaries
int2class_sex      = {key: value for key, value in enumerate(int2class_sex)}
int2class_title    = {key: value for key, value in enumerate(int2class_title)}
int2class_embarked = {key: value for key, value in enumerate(int2class_embarked)}

# These features are no longer necessary, let's drop them
df.drop(['Ticket', 'Cabin', 'Name'], axis=1, inplace=True)

# Let's store the categorial and continuous features in lists is good practice, it might come in handy in the future
categorical_features = ["Pclass","Sex","TravelsAlone","Title", "Embarked"]
continuous_features = ['Fare','Age','NameLength']

# Let's check if we forgot to fill up any missing value
print('Missing value percentage')
(df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)




_ = pd.concat([df, has_survided], axis=1).hist()
plt.tight_layout(pad=1)




int2class_embarked




# Feature distributions prior to StandardScaler
_ = df[continuous_features].hist()
plt.tight_layout(pad=1)




# Applying StandardScaler
for col in continuous_features:
    transf = df[col].values.reshape(-1,1)
    scaler = StandardScaler().fit(transf)
    df[col] = scaler.transform(transf)




# Feature distributions prior to StandardScaler
_ = df[continuous_features].hist()
plt.tight_layout(pad=1)




_, ax = plt.subplots(figsize=(12,8))
_ = sns.heatmap(pd.concat([df, has_survided], axis=1).corr(), annot=True, fmt=".1f", cbar_kws={'label': 'Percentage %'}, cmap="coolwarm", ax=ax)
_ = ax.set_title("Feature Correlation Matrix")

# Try out the interactive plot as alternative below! 
# pd.concat([df, has_survided], axis=1).corr().iplot(kind='heatmap', colorscale="RdBu", title="Feature Correlation Matrix") 




train_df = df.loc[train_index, :]
train_df['Survived'] = has_survided
test_df = df.loc[test_index, :]

del df




X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']




print("Target Variable Distribution - Survived")
print(y.value_counts(normalize=True))




# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
X_train.shape, y_train.shape, X_test.shape, y_test.shape




def eval_model(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    output_dict =  classification_report(y_test, pred, output_dict=True)
    fpr, tpr, _ = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
    output_dict['auc'] = skplt.metrics.auc(fpr, tpr)
    output_dict['classifier'] = model
    return output_dict




# I usually store every model and corresponding metric in dictionaries for quick later access
models = {}
models['KNN']                = eval_model(KNeighborsClassifier())
models['LogisticRegression'] = eval_model(LogisticRegression())
models['RandomForest']       = eval_model(RandomForestClassifier())

models = pd.DataFrame.from_dict(models)
for metric in ['precision', 'recall', 'f1-score']:
    models = models.append(
        models.loc['macro avg'].apply(
            lambda x: dict(x)[metric]).rename(f'{metric} avg'))




models.loc[['accuracy', 'auc', 'precision avg', 'recall avg', 'f1-score avg']].T.sort_values(by='accuracy', ascending=False).iplot('bar', title='Model Performance Comparison - Four Metrics', yrange=[.7,.85])




model = models['LogisticRegression']['classifier']
skplt.metrics.plot_confusion_matrix(y_test, model.predict(X_test), title='Logistic Regression Confusion Matrix')




model = models['LogisticRegression']['classifier']
skplt.metrics.plot_roc_curve(y_test, model.predict_proba(X_test))




model = models['RandomForest']['classifier']
feats = {feature: importance for feature, importance in zip(X.columns, model.feature_importances_)}
pd.DataFrame.from_dict(feats,orient='index', columns=['importance']).sort_values('importance').iplot('barh', title='Feature Importances - RandomForest')

