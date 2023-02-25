#!/usr/bin/env python
# coding: utf-8



"""
DO NOT edit the code below
"""

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier




"""
DO NOT edit the code below
"""

def plot_decision_boundary(X, model):
    h = .02
    
    x_min, x_max = X[:, 0].min() -1, X[:,0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    plt.contour(xx, yy, z, cmap=plt.cm.Paired)
    




np.random.seed(10)

N =   
D =    
X =    

delta = 1.5

X[:N//2] = 
X[N//2:] =    
Y = 

"""
DO NOT edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()

model =  
model.fit(X, Y)
print("score for basic tree:", model.score(X, Y))


plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 

plt.show()




model_depth_3 =  
#TODO 
print("score for basic tree:", model_depth_3.score(X, Y))




plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
plt.show()




model_depth_5 = 
#TODO 
print("score for basic tree:", model_depth_5.score(X, Y))




plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
plt.show()





np.random.seed(10) 
N =    
D =   
X =  

delta = 1.75

X[:125] = 
X[125:250] = 
X[250:375] = 
X[375:] = 
Y = 

"""
DO NOT edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()


#TODO

print("score for basic tree:", model.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
plt.show()


print("score for basic tree:", model_depth_3.score(X, Y))
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
plt.show()

#TODO
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
#TODO 
plt.show()




np.random.seed(10)  
N =  
D =   
X =   

delta = 1.75
X[:125] = 
X[125:250] = 
X[250:375] = 
X[375:] = 
Y = 

"""
DO NOT edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()

#TODO

#TODO

np.random.seed(10)

N = 500
D = 2
X = np.random.randn(N, D)

R_smaller =  
R_larger =   

R1 =     
theta =    
X[:250] = np.concatenate([[R1 * np.cos(theta)], [R1*np.sin(theta)]]).T 

R2 =     
theta =  
X[250:] = np.concatenate([[R2 * np.cos(theta)], [R2*np.sin(theta)]]).T 

Y =  

"""
Do Not edit code below
"""
plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
plt.show()

#TODO

#TODO






