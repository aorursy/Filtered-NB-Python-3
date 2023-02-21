import pandas as pd 
import numpy as np
from sklearn.linear_model import perceptron
from sklearn import cross_validation
from sknn.mlp import Classifier, Layer

#Print you can execute arbitrary python code
data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
#test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

# transform 'Sex' to an boolean isFemale
data['isFemale'] = data['Sex'].map({'female': 1, 'male':1})
data['isFemale'] = data['isFemale'].astype(int)

# transform 'Embarked' to an int 'intEmbarked'
data['intEmbarked'] = data['Embarked'].map({'S':1, 'C':2, 'Q':3}, na_action='ignore')
mode = data['intEmbarked'].mode().iloc[0]
data.loc[data['intEmbarked'].isnull(), 'intEmbarked'] = mode
data['intEmbarked'] = data['intEmbarked'].astype(int)

# fill missing values for the age with the mean
meanAge = data['Age'].mean()
data.loc[data['Age'].isnull(), 'Age'] = meanAge
data['Age'] = data['Age'].astype(int)

# pull labels out of training data
labels = data['Survived']

# drop string columns and lables
data = data.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Survived'], axis=1)

# seperate a training set
trainingset = data.values

print "Training perceptron..."
# train perceptron
pPenalty = None
pAlpha = 0.0001
pfit_intercept = True
pIterations = 100
pInfo = 'Perceptron Classifier:\nPenalty: ' + str(pPenalty) + '\nAlpha: ' + str(pAlpha) + '\nIterations: ' + str(pIterations) + '\n'

perceptron = perceptron.Perceptron(penalty=pPenalty,alpha=pAlpha,fit_intercept=pfit_intercept,n_iter=pIterations)
#perceptron.fit(trainingset[0::,1::], trainingset[0::,0])

# train multilayer perceptron
nLearningRate = 0.001
nIterations = 25

nn = Classifier(
        layers= [ Layer("Maxout", units=100, pieces=2),
                  Layer("Softmax")],
        leanring_rate= nLearningRate,
        n_iter= nIterations
        )

# get results for traing set
print "Getting scores..."
pScores = cross_validation.cross_val_score(perceptron, trainingset[0::, 1::], trainingset[0::,0], cv=5)
pScores = np.array(pScores)

strPScores = 'Average score: ' + str(pScores.mean()) + '\nScores:\n' + str(pScores) 

print pInfo + '\n' + strPScores

nScores = cross_validation.cross_val_score(nn, trainingset[0::, 1::], trainingset[0::,0], cv=5)
nScores = np.array(nScores)

strNScores = 'Average score: ' + str(nScores.mean()) + '\nScores:\n' + str(nScores) 

print '\n\nNueral Net results\n' + strNScores
