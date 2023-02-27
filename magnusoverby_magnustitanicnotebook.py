#!/usr/bin/env python
# coding: utf-8



import sklearn as sk #Machine learning stuff# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn import linear_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




titanic = pd.read_csv("../input/train.csv")
survived = pd.read_csv("../input/gendermodel.csv")
gender = pd.read_csv("../input/genderclassmodel.csv")
test_titanic = pd.read_csv("../input/test.csv")









#Encode female 1 is female 0 is Male
def EncodeSex(combined):
    combined["Sex"] = combined["Sex"].apply(lambda x:1 if x=="female" else 0)
    return combined
#Convert the whole row
#print(combined["Age"])




# Lets do a regression how well age explains Survived.... lets gradually add more variables later
# drop na for the intended variables
features = ["Age","Sex","Pclass","Survived","PassengerId"]
featuresExSurvived= ["Age","Sex","Pclass","PassengerId"] 
xfeatures = ["Age","Sex","Pclass"]
Ntitanic = EncodeSex(titanic.copy()).dropna()
print(Ntitanic[Ntitanic["Sex"]==1].head(2))
X = Ntitanic[xfeatures]
Y = Ntitanic["Survived"]




logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
logreg.score(X,Y)




test_titanic = EncodeSex(test_titanic)[featuresExSurvived].dropna()
X_pred = test_titanic[xfeatures]
Y_pred = logreg.predict(X_pred)














#Explore the data
#Fares not survived and fares survived in test date
# Titanic
ftitanic = titanic.copy()
Fares_survived = ftitanic[ftitanic["Survived"]==1]["Fare"].fillna(ftitanic["Fare"].median())
Fares_died = ftitanic[ftitanic["Survived"]==0]["Fare"].fillna(ftitanic["Fare"].median())
Fares_survived = Fares_survived.astype(int)
Fares_died = Fares_died.astype(int)




#plot survivers
plt.hist(Fares_survived,label="survived")
plt.hist(Fares_died,label="died")
plt.legend(loc = "upper right")
plt.show()




#Gender differances
gtitanic = EncodeSex(titanic.copy())
survived_males = gtitanic[gtitanic["Sex"]==0]["Survived"]
survived_females = gtitanic[gtitanic["Sex"]==1]["Survived"]
percentSurvived_males = np.sum(survived_males)/len(survived_males)
percentSurvived_females = np.sum(survived_females)/len(survived_females)
print(percentSurvived_males)
print(percentSurvived_females)
scale = [0.05*x for x in range(20)]
plt.bar([,[percentSurvived_males,percentSurvived_females]])
plt.show()




titstanic  = titanic[titanic["Sex"]=="female"]
titstanic.head(1)
