#!/usr/bin/env python
# coding: utf-8



import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from xgboost import XGBClassifier

get_ipython().run_line_magic('matplotlib', 'inline')

# helper funcs
sns.set(style="white", palette="muted", color_codes=True)
def is_nan(num):
    return num != num

# load data 
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
print("Total training examples: {n}".format(n=len(data_train)))
print("Total test examples: {n}".format(n=len(data_test)))

data_train.head(10)




data_test.loc[data_test['PassengerId'] == 1116]




# Pclass
pclass = data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
_ = sns.barplot(data=pclass, x='Pclass', y='Survived')




# Sex
sex = data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
_ = sns.barplot(data=sex, x='Sex', y='Survived')




# Age
cleaned_age = [c for c in data_train["Age"].values.tolist() if not is_nan(c)]  # some basic data cleaning
sns.distplot(cleaned_age, kde=False, color="b")
_ = sns.FacetGrid(data_train, col='Survived').map(plt.hist, 'Age', bins=20)




# Fare
cleaned_fare = [c for c in data_train["Fare"].values.tolist() if not is_nan(c)]  # some basic data cleaning
sns.distplot(cleaned_fare, kde=False, color="b")
_ = sns.FacetGrid(data_train, col='Survived').map(plt.hist, 'Fare', bins=20)




# Ticket last digit
data_train['Ticket_last_digit'] = data_train["Ticket"].map(lambda x: float(x[-1:]) % 10 if x[-1:].isdigit() else float('NaN'))
_ = sns.distplot([c for c in data_train["Ticket_last_digit"].values.tolist() if not is_nan(c)], kde=False, color="b")
del data_train['Ticket_last_digit']




# preprocess data
_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]

def sanity_checks(df, is_train):
    assert (not is_train) or all(x in [0, 1] for x in df.Survived)
    assert all(x in [1,2,3] for x in df.Pclass)    
    assert all(not is_nan(x) for x in df.SibSp)
    assert all(not is_nan(x) for x in df.Parch)
    assert all(x in ["S", "C", "U", "Q"] for x in df.Embarked)

def preprocess_sex(df):
    df.Sex = df.Sex.apply(lambda x: x.lower() if x.lower() in ["female", "male"] else "unknown")
    return df

def preprocess_age(df):
    # bucket age
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df
  
def preprocess_fare(df):
    # bucket fare
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 50, 100, 300, 1000)
    group_names = ['Unknown', 'fare_1', 'fare_2', 'fare_3', 'fare_4', 'fare_5', 'fare_6', 'fare_7']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def preprocess_cabin(df):
    # get init letter of cabin
    df.Cabin = df.Cabin.fillna('X')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def drop_features(df, features, is_train):
    if is_train:
        return df[features + ['PassengerId','Survived']]
    else:
        return df[features + ['PassengerId']]

def preprocess_embarked(df):
    df.Embarked = df.Embarked.fillna('U')
    return df

# run this func for both train and test
def preprocess1(df, is_train):    
    df = preprocess_embarked(df)
    df = preprocess_sex(df)
    df = preprocess_age(df)
    df = preprocess_fare(df)
    df = preprocess_cabin(df)
    df = drop_features(df, features=_features, is_train=is_train)
    sanity_checks(df, is_train)
    return df

data_train1 = preprocess1(data_train.copy(), is_train=True)
data_test1 = preprocess1(data_test.copy(), is_train=False)
data_train1.head()




# encode features
def encode_features(train_df, test_df, features):    
    combined = pd.concat([train_df[features], test_df[features]])
    
    encoders = list()
    for feature in features:
        le = preprocessing.LabelEncoder()
        le.fit(combined[feature])
        train_df[feature] = le.transform(train_df[feature])
        test_df[feature] = le.transform(test_df[feature])
        encoders.append(le)
    train_df = train_df.sort_values("PassengerId")
    test_df = test_df[['PassengerId'] + features].sort_values('PassengerId')
    assert(len(encoders) == len(features))
    return train_df, test_df, encoders
       
data_train1, data_test1, encoders = encode_features(data_train1, data_test1, features=_features)
print("First 5 data_train out of {n}".format(n=len(data_train1.values)))
print(data_train1.head(5))
print("First 5 data_test out of {n}".format(n=len(data_test1.values)))
print(data_test1.head(5))




data_train1.corr().sort_values(by='Survived', axis=0, ascending=False).Survived




X_train1 = data_train1[_features]
y_train1 = data_train1['Survived']
print("Total data train: {n}".format(n=len(data_train1.values)))
print("First 3 X_train out of {n}".format(n=len(X_train1.values)))
print(X_train1.head(3))
print("First 3 y_train out of {n}".format(n=len(y_train1.values)))
print(y_train1.head(3))




# First let's try default hyper-parameters of SVM.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from time import time
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import LinearSVC


X_svm_train, X_svm_test, y_svm_train, y_svm_test = train_test_split(X_train1, y_train1,
                                                                    test_size=0.2, random_state=42)
lsvc = LinearSVC()
lsvc.fit(X_svm_train, y_svm_train)
y_svm_predict = lsvc.predict(X_svm_test)
print("Default hyper-parameter. SVM accuracy = ", accuracy_score(y_svm_predict, y_svm_test))




def get_best_estimator(estimator, param, X_train, y_train, verbose=0, n_jobs=1, cv=5):
    """Run grid search to get the best hyper-parameter set for the given estimator.
    param is the map of various of hyper-parameters.
    cv is an int to specify the number of folds for training.
    """
    # Type of scoring used to compare parameter combinations
    acc_scorer = make_scorer(accuracy_score)

    grid_search = GridSearchCV(
        estimator=estimator, 
        param_grid=param,
        scoring=acc_scorer,
        verbose=verbose,
        cv=cv)

    start = time()
    grid_search.fit(X_train, y_train)
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))

    # get the best hyper-param set and its score
    return grid_search.best_estimator_, grid_search.best_score_




from sklearn.svm import LinearSVC
lsvc_param = {
    'C': np.logspace(-2, 2, 4),
    'dual': [False],
    'penalty': ['l1', 'l2']
}
lsvc, lsvc_accu = get_best_estimator(estimator=LinearSVC(), param=lsvc_param, X_train=X_train1, y_train=y_train1)
print("GridSearchCV found the best hyper-parameter set for LinearSVC:\n{s}\n\n{r}".format(
    s=lsvc_accu, r=lsvc.get_params()))
# re-train the model using the best hyper-param set on entire training set
lsvc.fit(X_train1, y_train1)
# test the model on test set
y_submit_predict = lsvc.predict(data_test1[_features])
submission = pd.DataFrame({
        "PassengerId": data_test1["PassengerId"],
        "Survived": y_submit_predict
    })
#submission.to_csv('../output/submission.csv', index=False)
submission.head(10)




from sklearn.linear_model import LogisticRegression

# specify the ranges of hyper-parameters
logreg_param = {
    #'penalty': ['l1', 'l2'],
    'penalty': ['l2'],
    #'C': np.logspace(-3, 4, 8),
    'C': [1.0],
}
logreg, logreg_accu = get_best_estimator(estimator=LogisticRegression(), param=logreg_param, X_train=X_train1, y_train=y_train1)
print("GridSearchCV found the best hyper-parameter set for LogReg:\n{s}\n\n{r}".format(
    s=logreg_accu, r=logreg.get_params()))




from sklearn.linear_model import Perceptron

# specify the ranges of hyper-parameters
percep_param = {
    #'penalty': ['l1', 'l2', 'elasticnet'],
    'penalty': ['l1'],
    #'alpha': np.logspace(-5, 2, 8),
    'alpha': [0.0001],
    #'n_iter': np.arange(1,8),
    'n_iter': [5]
}
percep, percep_accu = get_best_estimator(estimator=Perceptron(), param=percep_param, X_train=X_train1, y_train=y_train1)
print("GridSearchCV found the best hyper-parameter set for Percepton:\n{s}\n\n{r}".format(
    s=percep_accu, r=percep.get_params()))




from sklearn.neighbors import KNeighborsClassifier

knn_param = {
    #'n_neighbors': np.arange(1, 10),
    'n_neighbors': [5],
    #'weights': ['uniform', 'distance'],
    'weights': ['uniform'],
    #'p': [1, 2],
    'p': [1],
}
knn, knn_accu = get_best_estimator(estimator=KNeighborsClassifier(), param=knn_param, X_train=X_train1, y_train=y_train1)
print("GridSearchCV found the best hyper-parameter set for KNN:\n{s}\n\n{r}".format(
    s=knn_accu, r=knn.get_params()))




from sklearn.svm import SVC
svc_param = {
    #'C': np.logspace(-2, 2, 4),
    'C': [4.6415888336127775],
    #'kernel': ['rbf', 'linear'],
    'kernel': ['rbf'],
}
svc, svc_accu = get_best_estimator(estimator=SVC(), param=svc_param, X_train=X_train1, y_train=y_train1)
print("GridSearchCV found the best hyper-parameter set for SVC:\n{s}\n\n{r}".format(
    s=svc_accu, r=svc.get_params()))




from sklearn.tree import DecisionTreeClassifier

tree_param = {
    #'criterion': ['gini', 'entropy'],
    'criterion': ['entropy'],
    #'splitter': ['best', 'random'],
    'splitter': ['best'],
    #'max_features': [None, 'sqrt', 'log2'],
    'max_features': [None],
}

tree, tree_accu = get_best_estimator(estimator=DecisionTreeClassifier(), param=tree_param, X_train=X_train1, y_train=y_train1)
print("GridSearchCV found the best hyper-parameter set for DecisionTreeClassifier:\n{s}\n\n{r}".format(
    s=tree_accu, r=tree.get_params()))




from sklearn.ensemble import RandomForestClassifier

rf_param = {
    #'n_estimators': np.arange(5, 15, 1),
    'n_estimators': [11],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2'],
}

rf, rf_accu = get_best_estimator(estimator=RandomForestClassifier(), param=rf_param, X_train=X_train1, y_train=y_train1)
print("GridSearchCV found the best hyper-parameter set for RandomForestClassifier:\n{s}\n\n{r}".format(
    s=rf_accu, r=rf.get_params()))




# running the grid search on XGBoost takes long time. 
# I commented the ranges and set the best paramters learned.
xgb_param = {
    #'max_depth': np.arange(4, 10, 1),
    'max_depth': [4],
    #'learning_rate': np.logspace(-4, 1, 20),
    'learning_rate': [0.88586679041008232],
    #'n_estimators': np.arange(15, 16, 1),
    'n_estimators': [15],
    #'gamma': np.logspace(-4, 1, 20)
    'gamma': [2.9763514416313193]
}
xgb1, xgb_accu = get_best_estimator(estimator=XGBClassifier(), param=xgb_param, X_train=X_train1, y_train=y_train1, verbose=1, n_jobs=4)
print("GridSearchCV found the best hyper-parameter set for XGBClassifier:\n{s}\n\n{r}".format(
    s=xgb_accu, r=xgb1.get_params()))




m = pd.DataFrame({
    'Model': ['SVC', 'KNN', 'Logistic Regression', 
              'Random Forest', 'XGBoost', 'Perceptron', 
              'Linear SVC', 'Decision Tree'],
    'Score': [svc_accu, knn_accu, logreg_accu, 
              rf_accu, xgb_accu, percep_accu, 
              lsvc_accu, tree_accu]})
m.sort_values(by='Score', ascending=False)




from xgboost import plot_importance as xgb_plot_imp, plot_tree as xgb_plot_tree
_, ax = plt.subplots(1, 1, figsize=(7, 7))
_ = xgb_plot_imp(booster=xgb1, ax=ax)




_, ax = plt.subplots(1, 1, figsize=(12, 12))
_ = xgb_plot_tree(booster=xgb1, num_trees=1, ax=ax)
sex_coder = encoders[_features.index("Sex")]
print(list(zip(sex_coder.classes_, sex_coder.transform(sex_coder.classes_))))
pclass_coder = encoders[_features.index("Pclass")]
print(list(zip(pclass_coder.classes_, pclass_coder.transform(pclass_coder.classes_))))
age_coder = encoders[_features.index("Age")]
print(list(zip(age_coder.classes_, age_coder.transform(age_coder.classes_))))
cabin_coder = encoders[_features.index("Cabin")]
print(list(zip(cabin_coder.classes_, cabin_coder.transform(cabin_coder.classes_))))




# xgb1 was trained on 80% of training data (using cv)
# now retrain on the entire traning data.
xgb1.fit(X_train1, y_train1)
y_submit_predict = xgb1.predict(data_test1[_features])
submission = pd.DataFrame({
        "PassengerId": data_test1["PassengerId"],
        "Survived": y_submit_predict
    })
#submission.to_csv('../output/submission.csv', index=False)
submission.head(10)




submission.loc[submission['PassengerId'] == 1116]




from sklearn.model_selection import train_test_split
X_dbg_train, X_dbg, y_dbg_train, y_dbg = train_test_split(X_train1, y_train1,
                                                          test_size=0.2, random_state=42)

xgb1.fit(X_dbg_train, y_dbg_train)
y_predict = xgb1.predict(X_dbg)
all_predictions = zip(y_dbg.index.tolist(), y_dbg.tolist(), y_predict.tolist())
# Let's look at false negative: those who survived but model says they died
false_predictions = [t[0] for t in all_predictions if (t[1] == 1 and t[2] == 0)]

data_train.loc[data_train.index.isin(false_predictions)].loc[data_train.Sex == "male"].head(1)

