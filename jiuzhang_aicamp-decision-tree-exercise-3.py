#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score, train_test_split
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

data = #TODO
data.head()

data['safe_loans'].value_counts()

# use the percentage of bad and good loans to undersample the safe loans.
# bad_ones = # TODO
# safe_ones = # TODO
# percentage = #TODO

# risky_loans = bad_ones
# safe_loans = #TODO

# combine two kinds of loans
# data_set = #TODO


data_set[target].value_counts(normalize=True)


def label_encode(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data


#grade, home_ownership, target
cols = ['grade', 'term','home_ownership', 'emp_length']
# data_set = #TODO
data_set.head()


# train_data, test_data = #TODO
# trainX, trainY = #TODO
testX, testY = test_data[test_data.columns[1:]], pd.DataFrame(test_data[target])



def count_errors(labels_in_node):
    """
    Input: (Numpy Array/Pandas series)labels in node, eg: [-1,-1,1,-1,1]
    Output: (Int) if we do the major class voting, how many errors we make?
    """
    if len(labels_in_node) == 0:
        return 0
    
    # positive_ones = #TODO
    # negative_ones = #TODO
    
    # return # TODO


def best_split(data, features, target):
    """
    We want to select out the best feature such that it splits the data best based on your measurement(IG/accuracy)
    Input: (Pandas DataFrame)data
           (List of String) features  candidates we can choose feature from
           (String) target  the target name we shoot for. eg: 'safe_loan' 
           
    Output: (String) the best feature
    """
    # return the best feature
    best_feature = None
    best_error = 2.0 
    num_data_points = float(len(data))  

    for feature in features:
        
        # left_split = # TODO
        
        # right_split = #TODO
        
        # left_misses = #TODO            

        # right_misses = #TODO
            
        # error = #TODO

        # if error < best_error:
        #     best_error = #TODO
        #     best_feature = #TODO
    return best_feature




def entropy(labels_in_node):
    n = len(labels_in_node)
    s1 = (labels_in_node==1).sum()
    if s1 == 0 or s1 == n: # indicates the labels are the same~
        return 0
    
    p1 = float(s1) / n
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1)


def best_split_entropy(data, features, target):
    """
    We want to select out the best feature such that it splits the data best based on your measurement(IG/accuracy)
    Input: (Pandas DataFrame)data
           (List of String) features  candidates we can choose feature from
           (String) target  the target name we shoot for. eg: 'safe_loan' 
           
    Output: (String) the best feature
    """
    
    best_feature = None
    best_info_gain = float('-inf') 
    num_data_points = float(len(data))
    # entropy_original = #TODO

    for feature in features:
        
        # left_split = #TODO
        
        # right_split = #TODO 
        
        # left_entropy = #TODO           

        # right_entropy = #TODO
            
        # entropy_split = #TODO
        
        # info_gain = #TODO

        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature
    




# class TreeNode:
#     def __init__(self, is_leaf, prediction, split_feature):
#     # TODO


from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
class MyDecisionTree(BaseEstimator):
    
    def __init__(self, max_depth, min_error):
        self.max_depth = max_depth
        self.min_error = min_error
    
    def fit(self, X, Y, data_weights = None):
        
        data_set = pd.concat([X, Y], axis=1)
        # features = #TODO
        target = Y.columns[0]
        # self.root_node = # TODO
        
        
    def predict(self, X):
        """
        Input:  (Pandas DataFrame/Numpy array, size: m * n) a matrix and each row indicates a data point
        Output: (Pandas DataFrame/Numpy array, size: m * 1) array of the predicted result
        
        Tips: each row is predicted by the function predict_single_data()
        """
        prediction = X.apply(lambda row: self.predict_single_data(self.root_node, row), axis=1)
        return prediction
        
        
    def score(self, testX, testY):
        """
        Tips: use defined predict function to get predicted result and compare it with testY
        """
        # target = # TODO  the target name
        result = self.predict(testX)
        return accuracy_score(testY[target], result)
    
    
    def create_tree(self, data, features, target, current_depth = 0, max_depth = 10, min_error=0):
        """
        Input
            data: (pandas data frame) the input data
            features: (pandas series/dataframe/numpy array) available features
            target: (pandas series/dataframe/numpy array)  the target to predict
            current_depth: (Int)  current depth of the tree
            max_depth: (Int)  the maximum depth of the tree
            min_error: (Float) the minimum error reduction  
            
        Output:
            (TreeNode)  root
        """        
        

        # remaining_features = #TODO

        # target_values = # TODO

        # termination 1   bonus task
        #if count_errors(target_values) <= min_error:
        #    print("Termination 1 reached.")     
        #    return # TODO

        # termination 2
        if len(remaining_features) == 0:
            print("Termination 2 reached.")    
            return #TODO    

        # termination 3
        if current_depth >= max_depth: 
            print("Termination 3 reached.")
            return #TODO




        #split_feature = # TODO 
        # split_feature = # TODO 

        # left_split = # TODO
        # right_split = # TODO

        # remaining_features = # TODO
        # print("Split on feature %s. (%s, %s)" % (split_feature, str(len(left_split)), str(len(right_split))))

        if len(left_split) == len(data):
            print("Perfect split!")
            return self.create_leaf(left_split[target])
        if len(right_split) == len(data):
            print("Perfect split!")
            return self.create_leaf(right_split[target])

        # left_tree = # TODO     
        # right_tree = # TODO

        result_node = TreeNode(False, None, split_feature)
        result_node.left = left_tree
        result_node.right = right_tree
        return result_node    
    
    
    
    def create_leaf(self, target_values):
        """
        Input: (Pandas DataFrame/Numpy Array)  target_values  eg: [-1,1,-1,-1,1]
        Output: (TreeNode) node   Note that you should fill in the correct information for each attribute of the result
        """

        # leaf = # TODO

        # num_positive_ones = #TODO
        # num_negative_ones = #TODO

        if num_positive_ones > num_negative_ones:
            leaf.prediction = 1
        else:
            leaf.prediction = -1

        return leaf 
    
    
    
    def predict_single_data(self, tree, x, annotate = False): 
        """
        Input:  (TreeNode)  tree
                (Pandas DataFrame) x  it's a single array or one row from a pandas dataframe (one data point)
                (Bool)  annotate  if intermediate result is displayed
        Output:  (Int)  -1 or 1 in our case
        """
        
        if tree.is_leaf:
            if annotate: 
                # print("leaf node, predicting %s" % tree.prediction)
            return # TODO 
        else:
            split_feature_value = # TODO

            if annotate: 
                # print("Split on %s = %s" % (tree.split_feature, split_feature_value))
            if split_feature_value == 0:
                return # TODO
            else:
                return # TODO    
    
    def count_leaves(self):
        return self.count_leaves_helper(self.root_node)
    
    # def count_leaves_helper(self, tree):
    #     # TODO
    




m = MyDecisionTree(max_depth = 10, min_error = 1e-15)




m.fit(trainX, trainY)




m.score(testX, testY)




m.count_leaves()




# model_1 = # TODO
# model_2 = # TODO
# model_3 = # TODO




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






