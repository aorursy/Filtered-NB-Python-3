#!/usr/bin/env python
# coding: utf-8



import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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




import re

# preprocess data
_name_tokens = ['mr', 'ms', 'miss', 'dr', 'mrs']

def sanity_checks(df, is_train):
    assert (not is_train) or all(x in [0, 1] for x in df.Survived)
    assert all(x in [1,2,3] for x in df.Pclass)    
    assert all(not is_nan(x) for x in df.SibSp)
    assert all(not is_nan(x) for x in df.Parch)
    assert all(x in ["S", "C", "U", "Q"] for x in df.Embarked)

def extract_token_from_name(name):
    for ch in "(),.\'\"":
        name = name.replace(ch, " ")
    tokens = name.split()
    name = "unknown"
    for token in tokens:
        if token.lower() in _name_tokens:
            name = token.lower()
    return name

def preprocess_name(df):    
    df.Name = df.Name.apply(lambda x: extract_token_from_name(x))
    return df

def preprocess_sex(df):
    df.Sex = df.Sex.apply(lambda x: x.lower() if x.lower() in ["female", "male"] else "unknown")
    return df

def preprocess_age(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df
  
def preprocess_fare(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 50, 100, 300, 1000)
    group_names = ['Unknown', 'fare_1', 'fare_2', 'fare_3', 'fare_4', 'fare_5', 'fare_6', 'fare_7']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def cabin_to_range(cabin):    
    if is_nan(cabin):
        return 'X'
    r = re.match(r"(\w+ )*([A-Z])(\d+)", cabin)
    if r:
        return "%s%d" % (r.groups()[-2], int(r.groups()[-1]) / 4)
    else:
        return 'X'

def cabin_to_oddness(cabin):    
    if is_nan(cabin):
        return 'X'
    r = re.match(r"(\w+ )*([A-Z])(\d+)", cabin)
    if r:
        return "%s%d" % (r.groups()[-2], int(r.groups()[-1]) % 2)
    else:
        return 'X'

def preprocess_cabin(df):
    df['Cabin_range'] = df["Cabin"].apply(lambda x: cabin_to_range(x))
    df['Cabin_oddness'] = df["Cabin"].apply(lambda x: cabin_to_oddness(x))
    df = df.drop(['Cabin'], axis=1)
    return df

def drop_features(df):
    return df.drop(['Ticket'], axis=1)

def preprocess_embarked(df):
    df.Embarked = df.Embarked.fillna('U')
    return df

# run this func for both train and test
def preprocess(df, is_train):    
    df = preprocess_embarked(df)
    df = preprocess_sex(df)
    df = preprocess_age(df)
    df = preprocess_fare(df)
    df = preprocess_cabin(df)
    df = preprocess_name(df)
    df = drop_features(df)
    sanity_checks(df, is_train)
    return df


data_train = preprocess(data_train.copy(), is_train=True)
data_test = preprocess(data_test.copy(), is_train=False)
data_train.head()




# encode features
_features = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin_range", "Cabin_oddness", "Embarked"]

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
       
data_train, data_test, _ = encode_features(data_train, data_test, features=_features)
print("First 5 data_train out of {n}".format(n=len(data_train.values)))
print(data_train.head(5))
print("First 5 data_test out of {n}".format(n=len(data_test.values)))
print(data_test.head(5))




from sklearn.model_selection import KFold

def split_data_train(data_train, features, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=13)

    # training features
    X = data_train[features]
    
    # training labels
    y = data_train['Survived']
    
    print("Split train data into %d folds" % kf.get_n_splits(X))
    
    print("First 3 X_train out of {n}".format(n=len(X.values)))
    print(X.head(3))
    print("First 3 y_train out of {n}".format(n=len(y.values)))
    print(y.head(3))
    
    return X, y, kf

X, y, kf = split_data_train(data_train=data_train, features=_features)




from time import time
from sklearn.metrics import make_scorer, accuracy_score

BASE_MODELS = ['LogReg', 'Perceptron', 'KNN', 'SVC', 'LinearSVC', 'DecisionTree', 'RandomForest', 'XGBoost']
NUM_OF_BASE_MODELS = len(BASE_MODELS)
model_accuracy = dict.fromkeys(BASE_MODELS)

X_stack_train = pd.DataFrame(np.zeros((X.shape[0], NUM_OF_BASE_MODELS),
                                      dtype=np.float64),
                             columns=BASE_MODELS)
y_stack_train = y  # ground truth
X_stack_test = pd.DataFrame(np.zeros((data_test.shape[0], NUM_OF_BASE_MODELS),
                                     dtype=np.float64),
                            columns=BASE_MODELS)

def _get_accuracy(y_fold, y_predict):
    """helper function"""
    # y_predict may be a proba. convert it to 0/1
    y_tmp = np.copy(y_predict)
    for i in range(len(y_tmp)):
        y_tmp[i] = 1 if y_tmp[i] >= 0.5 else 0
    return accuracy_score(y_fold, y_tmp)    
    

def train_base_model_and_predict(estimator, X, y, model_name, kf, X_test, predictor='predict_proba'):
    """
    Input
    X: full train data
    y: full train label
    model_name: each base model has a unique name
    kf: KFolder spliter
    X_test: test data
    predictor: 'predict' or 'predict_proba'
    Output
    X_stack_train
    
    """
    model_idx = BASE_MODELS.index(model_name)
    print("Trainging Base Model #%d:" % model_idx)
    print(estimator)
    start = time()    
    # this stores n_folds this many predicts of test data. 
    n_folds = kf.get_n_splits(X)
    y_predict_of_model = np.zeros((data_test.shape[0], n_folds))
    # we call the prediction fold of training data as "stack_train"
    i = 0
    mean_accuracy = 0.0
    for train_index, stack_train_index in kf.split(X):
        # train model on X_train (4 folds)
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]        
        estimator.fit(X_train, y_train)
        
        # predict on the single fold
        X_fold = X.iloc[stack_train_index]
        y_fold = y.iloc[stack_train_index]
        y_stack_train_predict = estimator.predict_proba(X_fold)[:, 1] if predictor == 'predict_proba' else estimator.predict(X_fold)
        X_stack_train.iloc[stack_train_index, model_idx] = y_stack_train_predict 
        accuracy = _get_accuracy(y_fold=y_fold, y_predict=y_stack_train_predict)
        print("Accuracy = {s} on fold {i}".format(s=accuracy, i=i))
        mean_accuracy += accuracy
        
        # also predict on X_test
        y_predict_of_model[:, i] = estimator.predict_proba(X_test)[:, 1] if predictor == 'predict_proba' else estimator.predict(X_test)
        i += 1
    # finally average y_predict of test 
    X_stack_test.iloc[:, model_idx] = y_predict_of_model.mean(1)        
    print("Training took %.2f seconds" % (time() - start))
    model_accuracy[model_name] = mean_accuracy / i
    print("Mean accuracy = %.4f" % model_accuracy[model_name])




from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(penalty='l2', C=1.0)
train_base_model_and_predict(estimator=logreg, X=X, y=y, model_name='LogReg', kf=kf, 
                             X_test=data_test[_features], predictor='predict_proba')




from sklearn.linear_model import Perceptron

perceptron = Perceptron(n_iter=5, alpha=0.0001, penalty='l1')
train_base_model_and_predict(estimator=perceptron, X=X, y=y, model_name='Perceptron', kf=kf,
                             X_test=data_test[_features], predictor='predict')




from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', p=1)
train_base_model_and_predict(estimator=knn, X=X, y=y, model_name='KNN', kf=kf,
                             X_test=data_test[_features], predictor='predict_proba')




from sklearn.svm import SVC
svc = SVC(kernel='rbf', C=4.6415888336127775)
train_base_model_and_predict(estimator=svc, X=X, y=y, model_name='SVC', kf=kf,
                             X_test=data_test[_features], predictor='predict')




from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.21544346900318834, penalty='l1', dual=False)
train_base_model_and_predict(estimator=lsvc, X=X, y=y, model_name='LinearSVC', kf=kf,
                             X_test=data_test[_features], predictor='predict')




from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_features='sqrt')
train_base_model_and_predict(estimator=tree, X=X, y=y, model_name='DecisionTree', kf=kf,
                             X_test=data_test[_features], predictor='predict_proba')




from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=11, criterion='entropy', max_features=None)
train_base_model_and_predict(estimator=rf, X=X, y=y, model_name='RandomForest', kf=kf,
                             X_test=data_test[_features], predictor='predict_proba')




xgb = XGBClassifier(max_depth=5, learning_rate=0.48329302385717521, n_estimators=15, gamma=0.8858)
train_base_model_and_predict(estimator=xgb, X=X, y=y, model_name='XGBoost', kf=kf,
                             X_test=data_test[_features], predictor='predict')




import operator
print("== Model Accuracy ==")
for m in model_accuracy:
    print(m, model_accuracy[m])
print("\n== Model Correlations ==")
print(X_stack_train.corr())
_ = sns.heatmap(X_stack_train.corr(), vmin=0, vmax=1)




# level 2 model selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

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

def try_each_stacking_model_combo(candidates):
    # Try each base model combination
    stack_performance = dict.fromkeys(candidates)
    cnt = 1
    for c in candidates:
        print("\nTrying combination #%d: %s" % (cnt, c))
        cnt += 1
        X_stack_train_tmp = X_stack_train[list(c)]
        X_stack_test_tmp = X_stack_test[list(c)]

        # specify the ranges of hyper-parameters
        logreg_param = {
            'penalty': ['l1', 'l2'],
            'C': np.logspace(-3, 4, 8),
        }
        logreg_stacking, logreg_stacking_accu = get_best_estimator(
            estimator=LogisticRegression(), param=logreg_param, X_train=X_stack_train_tmp, y_train=y_stack_train)
        print("GridSearchCV found the best hyper-parameter set for LogReg_stacking:\n{s}\n\n{r}".format(
            s=logreg_stacking_accu, r=logreg_stacking.get_params()))
        stack_performance[c] = logreg_stacking_accu
    print("\nThe final best combinations:")
    print(sorted(stack_performance.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])
    
from itertools import combinations
STACKING_MODELS = list(combinations(BASE_MODELS, r=2)) + list(combinations(BASE_MODELS, r=3)) + list(combinations(BASE_MODELS, r=4)) 
print("Tring total %d combinations" % len(list(STACKING_MODELS)))
# Note: uncomment this if you want to re-run the tuning.
#try_each_stacking_model_combo(candidates=STACKING_MODELS)

# The final best combinations:
#[(('SVC', 'LinearSVC', 'RandomForest', 'XGBoost'), 0.83052749719416386), 
#(('LogReg', 'LinearSVC', 'DecisionTree', 'RandomForest'), 0.82940516273849607), 
#(('LogReg', 'LinearSVC', 'RandomForest', 'XGBoost'), 0.82940516273849607), 
#(('LogReg', 'SVC', 'LinearSVC', 'RandomForest'), 0.82940516273849607), 
#(('LogReg', 'KNN', 'LinearSVC', 'RandomForest'), 0.82940516273849607)]




FINAL_BASE_MODELS = ["SVC", "LinearSVC", "RandomForest", "XGBoost"]
X_stack_train = X_stack_train[FINAL_BASE_MODELS]
X_stack_test = X_stack_test[FINAL_BASE_MODELS]
print(X_stack_train.head(5))
print(X_stack_test.head(5))
print("\n== Model Correlations ==")
print(X_stack_train.corr())
_ = sns.heatmap(X_stack_train.corr(), vmin=0, vmax=1)




def run_best_stacking_model(final_stacking_models):
    print("Trying ", final_stacking_models)
    X_stack_train_tmp = X_stack_train[final_stacking_models]
    X_stack_test_tmp = X_stack_test[final_stacking_models]

    # specify the ranges of hyper-parameters
    logreg_param = {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-3, 4, 8),
    }
    logreg_stacking, logreg_stacking_accu = get_best_estimator(
        estimator=LogisticRegression(), param=logreg_param, X_train=X_stack_train_tmp, y_train=y_stack_train)
    print("GridSearchCV found the best hyper-parameter set for LogReg_stacking:\n{s}\n\n{r}".format(
        s=logreg_stacking_accu, r=logreg_stacking.get_params()))
    
    # now we've found the best hyper-param using CV, re-train the level 2 model on all stack_train data
    logreg_stacking.fit(X_stack_train_tmp, y_stack_train)
    # predict the test data
    y_submit_predict = logreg_stacking.predict(X_stack_test_tmp)
    submission = pd.DataFrame({
            "PassengerId": data_test["PassengerId"],
            "Survived": y_submit_predict
        })
    #submission.to_csv('../output/submission.csv', index=False)
    submission.head(10)

run_best_stacking_model(FINAL_BASE_MODELS)

