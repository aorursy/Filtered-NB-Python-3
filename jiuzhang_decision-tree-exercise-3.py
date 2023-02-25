#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import tree
from sklearn import metrics

#data = pd.read_csv(os.path.join("data", "loan_sub.csv"), sep=',')
data = pd.read_csv(os.path.join("../input", "loan_sub.csv"), sep=',')

data.columns

# safe_loans =  1 => safe
# safe_loans = -1 => risky
#TODO

data['safe_loans'].value_counts(normalize=True)

cols = ['grade', 'term','home_ownership', 'emp_length']
target = 'safe_loans'

data.head()

data['safe_loans'].value_counts()

# use the percentage of bad and good loans to undersample the safe loans.


risky_loans = bad_ones


# combine two kinds of loans


data_set[target].value_counts(normalize=True)




def dummies(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data




#grade, home_ownership, target
cols = ['grade', 'term','home_ownership', 'emp_length']

data_set.head()





testX, testY = test_data[test_data.columns[1:]], pd.DataFrame(test_data[target])


def count_errors(labels_in_node):
    if len(labels_in_node) == 0:
        return 0
    



def best_split(data, features, target):
    # return the best feature
    best_feature = None
    best_error = 2.0 
    num_data_points = float(len(data))  

def entropy(labels_in_node):
    n = len(labels_in_node)
    s1 = (labels_in_node==1).sum()
    if s1 == 0 or s1 == n:
        return 0
    
    p1 = float(s1) / n
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def best_split_entropy(data, features, target):
    
    best_feature = None
    best_info_gain = float('-inf') 
    num_data_points = float(len(data))


    for feature in features:
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature      

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
class MyDecisionTree(BaseEstimator):
    
    def __init__(self, max_depth, min_error):
        self.max_depth = max_depth
        self.min_error = min_error
    
    def fit(self, X, Y, data_weights = None):
        
        data_set = pd.concat([X, Y], axis=1)
        target = Y.columns[0]
    
        
    def predict(self, X):
        prediction = X.apply(lambda row: self.predict_single_data(self.root_node, row), axis=1)
        return prediction
        
        
    def score(self, testX, testY):
        result = self.predict(testX)
        return accuracy_score(testY[target], result)
        
    def create_tree(self, data, features, target, current_depth = 0, max_depth = 10, min_error=0):
        # termination 1
        if count_errors(target_values) <= min_error:
            print("Termination 1 reached.")     

        # termination 2
        if len(remaining_features) == 0:
            print("Termination 2 reached.")    

        # termination 3
        if current_depth >= max_depth: 
            print("Termination 3 reached.")


        if len(left_split) == len(data):
            print("Perfect split!")
            return self.create_leaf(left_split[target])''
        if len(right_split) == len(data):
            print("Perfect split!")
            return self.create_leaf(right_split[target])

        result_node = TreeNode(False, None, split_feature)
        result_node.left = left_tree
        result_node.right = right_tree
        return result_node     
    
    def create_leaf(self, target_values):
        if num_positive_ones > num_negative_ones:
            leaf.prediction = 1
        else:
            leaf.prediction = -1
        return leaf 
    
    def count_leaves(self):
        return self.count_leaves_helper(self.root_node)

m = MyDecisionTree(max_depth = 10, min_error = 1e-15)

m.fit(trainX, trainY)

m.score(testX, testY)

m.count_leaves()


model_1.fit(trainX, trainY)
model_2.fit(trainX, trainY)
model_3.fit(trainX, trainY)

print("model_1 training accuracy :", model_1.score(trainX, trainY))
print("model_2 training accuracy :", model_2.score(trainX, trainY))
print("model_3 training accuracy :", model_3.score(trainX, trainY))

print("model_1 testing accuracy :", model_1.score(testX, testY))
print("model_2 testing accuracy :", model_2.score(testX, testY))
print("model_3 testing accuracy :", model_3.score(testX, testY))

print("model_1 complexity is: ", model_1.count_leaves())
print("model_2 complexity is: ", model_2.count_leaves())
print("model_3 complexity is: ", model_3.count_leaves())