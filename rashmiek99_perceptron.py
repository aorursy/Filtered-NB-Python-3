#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')




df = pd.read_csv('/kaggle/input/wineuci/Wine.csv',header=None)
#df = pd.read_csv('/kaggle/input/iris-dataset/iris.data.csv',header=None)

df.columns = [  'name'
                 ,'alcohol'
             ,'malicAcid'
             ,'ash'
            ,'ashalcalinity'
             ,'magnesium'
            ,'totalPhenols'
             ,'flavanoids'
             ,'nonFlavanoidPhenols'
             ,'proanthocyanins'
            ,'colorIntensity'
            ,'hue'
             ,'od280_od315'
             ,'proline'
                ]
df.head()




df.describe()




df.info()




corr = df[df.columns].corr()
sns.heatmap(corr, cmap="YlGnBu", annot = True);




print("Perceptron Model:\n")
print("We will split the data into train and test.The model is trained with train dataset and weights and bias are estimated.This weight and bias is used to evaluate the test dataset")




train,test = train_test_split(df,test_size=0.3)

scaler = StandardScaler()
encode = LabelEncoder()

X= train.drop(['name','ash'], axis=1).values
X = scaler.fit_transform(X)

y = train.iloc[:,0].values
y = encode.fit_transform(y)

final_X=  test.drop(['name','ash'], axis=1).values
final_X = scaler.fit_transform(final_X)

final_y = test.iloc[:,0].values
final_y = encode.fit_transform(final_y)




print("Activation function add non linearity to the function. It is required to make the network more    powerful and helps learn something complex.")

print("\nMost popular kinds of activation are :")

print("\n1. Sigmoid or Logistic - It is a S shaped curve with the formula :             \n\tf(x) = 1/1 + exp(-x)        \n\tThe range is between 0 and 1")

print("\n2. Tanh        \n\tf(x) = 1 - exp(-2x) /1 + exp(-2x)         \n\tThe range is between -1 and 1")

print("\n3. Relu - Rectified Linear Unit       \n\tR(x) = max(0,x)        \n\tThe values are 0 or x")




# We are using Relu activation 
def activation(t):
     return(np.where(t>=0,t,0))




print("Make prediction by calculating-        \n\t1. t = X*W + b        \n\t2. activation(t)")




def make_prediction(X,W,b):
    return activation(np.dot(X,W) + b)




print("Weights and bias are estimated for the training dataset using Stochastic Gradient Descent (SGD).\nSGD uses two parameters- \n1.learning rate:amount with which the weight & bias gets updated each time\n2.epoch:number of loops to run though the training data while updating the weight & bias\n\tw = w + (pred - target) * learning rate * training data\n\tb = b + (pred - target) * learning rate\nThe weights & bias are initialized to random small numbers or 0's and 1 respectively(as in our case)")




def train_weights(X,y,lr,epoch):
    
    W = np.zeros(X.shape[1])
    b = np.ones(1)
    
    for _ in range(epoch):
        for xi,target in zip(X,y):    
            
            y_pred = make_prediction(xi,W,b)
        
            adjustment = lr *(target - y_pred)

            W += (adjustment * xi)
            b += adjustment
    return W,b




print("Evaluate the model on test data using the weights and bias estimated in training process.\nCalculate the accuracy of the model")




def evaluate_model(X,y,W,b):
    
    y_true = make_prediction(X,W,b)
    
    y_true = np.round(y_true,0)
    y_true = y_true.astype(int)
    
    count = 0
    for i in range(len(y)):
        if y_true[i] == y[i]:
            count+=1
    
    return round((count/len(y) * 100),2)




W,b = train_weights(X,y,lr=0.01,epoch=1000)

acc = evaluate_model(final_X,final_y,W,b)

print("Accuracy = " + str(acc) + "%")

