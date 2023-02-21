# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import scipy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
genderclassmodel = pd.read_csv("../input/genderclassmodel.csv", delimiter=',')
gendermodel = pd.read_csv("../input/gendermodel.csv", delimiter=',')
test = pd.read_csv("../input/test.csv", delimiter=',')
train = pd.read_csv("../input/train.csv", delimiter=',')

def sigmoid(X):
    return 1./(1+np.exp(-X))

def compute_cost(theta,X,y): #computes cost given predicted and actual values
    m = X.shape[0] #number of training examples
    theta = np.reshape(theta,(len(theta),1))

#     y = reshape(y,(len(y),1))
    J = (1./m) * (-np.transpose(y).dot(np.log(sigmoid(X.dot(theta)))) - np.transpose(1-y).dot(np.log(1-sigmoid(X.dot(theta)))))
    
    grad = np.transpose((1./m)*np.transpose(sigmoid(X.dot(theta)) - y).dot(X))
    #optimize.fmin expects a single value, so cannot return grad
    return J[0][0]#,grad

#X = np.array([[1.,8.,1.,6.],[1.,3.,5.,7.],[1.,4.,9.,2.]])
#y = np.array([1., 0., 1.])[np.newaxis].T
#theta = np.array([-2., -1., 1., 2.])[np.newaxis].T

#print(compute_cost(theta, X, y))

#print(train['PassengerId'][:])
print 1



