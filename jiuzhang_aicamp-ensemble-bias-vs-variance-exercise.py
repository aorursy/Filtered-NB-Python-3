#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 引入mean_square_error
#TODO




# 数据集个数
#TODO
Num_datasets = 
noise_level = 

# 最大的degree
#TODO
max_degree =

# 每个数据集里的数据个数
#TODO
N = 

# 用于训练的数据数
#TODO
trainN = 
np.random.seed(2)




def make_poly(x, degree):
    """
    input: x  N by 1
    output: N by degree + 1
    """
    N = len(x)
    result = np.empty((N, degree+1))
    for d in range(degree + 1):
        result[:,d] = x ** d
        if d > 1:
            result[:, d] = (result[:, d] - result[:, d].mean()) / result[:,d].std()
    return result

def f(X):
    #TODO
    




x_axis = #TODO
y_axis = #TODO

# 可视化
#TODO




# 基本训练集
# TODO
X = 
np.random.shuffle(X)
f_X = f(X)

# 创建全部的数据
allData = make_poly(X, max_degree)

train_scores = np.zeros((Num_datasets, max_degree))
test_scores = np.zeros((Num_datasets, max_degree))

train_predictions = np.zeros((trainN, Num_datasets, max_degree))
prediction_curves = np.zeros((100, Num_datasets, max_degree))

model = LinearRegression()




# TODO




for k in range(Num_datasets):
    
    # 每个数据集不失pattern的情况下稍微不一样~
    Y = 
    
    trainX, testX = 
    trainY, testY = 
    
    # 用不同的模型去学习当前数据集
    for d in range(max_degree):
        
        # 模型学习
        # TODO
        
        
        # 在allData上的预测结果
        all_predictions = model.predict(allData[:, :d+2])
        
        # 预测并记录一下我们的目标函数
        x_axis_poly = make_poly(x_axis, d + 1)   # true poly x
        axis_predictions = model.predict(x_axis_poly)   # true y
        prediction_curves[:, k, d] = axis_predictions
        
        train_prediction = all_predictions[:trainN]
        test_prediction = all_predictions[trainN:]
        
        train_predictions[:, k, d] = train_prediction # 用于计算bias and varaince 
        
        
        #计算并存储训练集和测试集上的分数
        train_score = #TODO
        test_score = #TODO
        train_scores[k, d] = train_score
        test_scores[k, d] = test_score       
    
    




for d in range(max_degree):
    for k in range(Num_datasets):
        # TODO
        # 给定当前模型，画出它在所有数据集上的表现
        
        
    
    # TODO
    # 给定当前模型，画出它在所有数据集上的表现的平均
    
    
    plt.title("curves for degree=%d" %(d+1))
    plt.show()




#TODO 每一个模型的bias
average_train_prediction =    # 模型的平均表现
squared_bias = 

trueY_train = f_X[:trainN]# 真值




#TODO
for d in range(max_degree):
    for i in range(trainN):
        average_train_prediction[i,d] = 
    squared_bias[d] = 
        
        




variances = np.zeros((trainN, max_degree))
for d in range(max_degree):
    for i in range(trainN):
        #TODO
        difference = 
        variances[i,d] = 


# TODO
variance = 




degrees = np.arange(max_degree) + 1
best_degree = np.argmin(test_scores.mean(axis=0)) + 1




#TODO












