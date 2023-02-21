#!/usr/bin/env python
# coding: utf-8



# 必要的引入
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




data = 




# TODO 打印数据基本信息




# TODO 观察部分数据的形式
data.head(3)




# TODO 观察预测目标的分布




#TODO 可视化预测目标的分布




#TODO 利用sns画出每种舱对应的幸存与遇难人数




# TODO 打印部分名字信息




data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])




# TODO 打印name title信息




# TODO 名字title 与幸存的关系




# TODO 新增名字长度的变量




# TODO 名字长度与幸存的关系




# TODO 打印性别比例




# TODO 性别与幸存的关系




# TODO 年龄与幸存的关系




# TODO 登船地点的分布




# TODO 登船地点与幸存的关系




# TODO 可视化登船地点与舱位的关系




data['survived'].groupby(data['home.dest'].apply(lambda x: str(x).split(',')[-1])).mean()




def name(data):
    data['name_len'] = data['name'].apply(lambda x: len(x))
    data['name_title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
    del data['name']
    return data

def age(data):
    data['age_flag'] = data['age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    grouped_age = data.groupby(['name_title', 'pclass'])['age']
    data['age'] = grouped_age.transform(lambda x: x.fillna(data['age'].mean()) if pd.isnull(x.mean()) else x.fillna(x.mean()))
    return data

def embark(data):
    data['embarked'] = data['embarked'].fillna('Southampton')
    return data


def dummies(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data




# TODO 
# 去掉row.names, home.dest, room, ticket, boat等属性
drop_columns = 
data = 
data.head()




# TODO
# 利用name(), age(), embark(), dummies()等函数对数据进行变换
data = 
data = 
data = 
data = 
data.head()




from sklearn.model_selection import train_test_split
from sklearn import tree

#TODO 准备训练集合测试集， 测试集大小为0.2， 随机种子为33
trainX, testX, trainY, testY = 

# TODO 创建深度为3，叶子节点数不超过5的决策树
model = 
model.fit(trainX, trainY)




from sklearn import metrics
def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    #TODO 完成你自己的measure_performance函数

    if show_accuracy:
        #TODO
    
    if show_classification_report:
        #TODO
    
    if show_confussion_matrix:
        #TODO




# TODO 调用measure_performance 观察模型在testX, testY上的表现




# 利用 age, sex_male, sex_female做训练
sub_columns = 
sub_trainX = trainX[sub_columns]
sub_testX = testX[sub_columns]
sub_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
sub_model.fit(sub_trainX, trainY)




measure_performance(sub_testX, testY, sub_model)




import graphviz

dot_data = tree.export_graphviz(model, out_file=None, feature_names=trainX.columns) 

#TODO 生成graph文件
graph =  
#graph.render("titanic") 
#graph.view()
graph




# TODO 观察前20个特征的重要性
















