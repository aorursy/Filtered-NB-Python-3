#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotting libs
import seaborn as sns
import matplotlib.pyplot as plt

#sklearn lib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




hd = pd.read_csv("../input/heart-disease-uci/heart.csv")
print(hd.head())
print(hd.shape)




print("Missing data: \n")
print(str(hd.isnull().sum()))




labels=["Male", "Female"] #x-axis label
male = [hd[(hd["target"]==0)&(hd["sex"]==1)]["target"].count(), hd[(hd["target"]==0)&(hd["sex"]==0)]["target"].count()] #bars for males
female = [hd[(hd["target"]==1)&(hd["sex"]==1)]["target"].count(), hd[(hd["target"]==1)&(hd["sex"]==0)]["target"].count()] #bars for females
print(male)
print(female)
x = np.arange(len(labels)) #label locations
width=0.35 #bar widths

sumM = male[0]+female[0]
sumF = male[1]+female[1]

relm = [male[0]/sumM, male[1]/sumF]
relf = [female[0]/sumM, female[1]/sumF]

fig = plt.figure(figsize=(14,8))
ax = fig.subplots()
rects1 = ax.bar(x - width/2, male, width, label='Male')
rects2 = ax.bar(x + width/2, female, width, label='Female')
print(male[0]+female[0])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(["no disease", "disease"])

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()




hd["age_groups"] = hd["age"].apply(lambda x: 0 if x<6 else (1 if x < 18 else (2 if x < 30 else (3 if x < 50 else (4 if x < 65 else 5)))))
hd




fig = plt.figure(figsize=(14,8))
sns.distplot(hd["age"])




fig = plt.figure(figsize=(14,8))
sns.countplot(hd["age_groups"], hue=hd["target"])
print(hd["age_groups"].value_counts())




print(hd["fbs"].value_counts())




fig = plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.countplot(hd[hd["sex"]==1]["fbs"], hue=hd["target"])
plt.title("Male")
plt.gca().set_ylim(0,100)

plt.subplot(1,2,2)
sns.countplot(hd[hd["sex"]==0]["fbs"], hue=hd["target"])
plt.title("Female")
plt.gca().set_ylim(0,100)




print("fbs==0: {} % \nfbs==1: {} %".format(hd[hd["fbs"]==0]["fbs"].count()/len(hd["fbs"]), hd[hd["fbs"]==1]["fbs"].count()/len(hd["fbs"])))




fig = plt.figure(figsize=(14,6))
sns.countplot(hd["exang"], hue=hd["target"])




fig = plt.figure(figsize=(14,6))
plt.subplot(2,1,1)
sns.boxplot(hd[hd["target"]==0]["oldpeak"])
plt.xlim(0, 8)
plt.title("no disease")

plt.subplot(2,1,2)
sns.boxplot(hd[hd["target"]==1]["oldpeak"])
plt.xlim(0, 8)
plt.title("has disease")
plt.subplots_adjust(hspace=0.4)




fig = plt.figure(figsize=(14,6))
sns.countplot(hd["cp"], hue=hd["target"])




fig = plt.figure(figsize=(14,6))
plt.subplot(2,1,1)
sns.boxplot(hd[hd["target"]==0]["chol"])
plt.xlim(0, 600)
plt.title("no disease")

plt.subplot(2,1,2)
sns.boxplot(hd[hd["target"]==1]["chol"])
plt.xlim(0, 600)
plt.title("has disease")
plt.subplots_adjust(hspace=0.4)




fig = plt.figure(figsize=(14,6))
plt.subplot(2,1,1)
sns.boxplot(hd[hd["target"]==0]["thalach"])
plt.xlim(60, 210)
plt.title("no disease")

plt.subplot(2,1,2)
sns.boxplot(hd[hd["target"]==1]["thalach"])
plt.xlim(60, 210)
plt.title("has disease")
plt.subplots_adjust(hspace=0.4)




fig = plt.figure(figsize=(14,6))
plt.subplot(2,1,1)
sns.boxplot(hd[hd["target"]==0]["trestbps"])
plt.xlim(90, 210)
plt.title("no disease")

plt.subplot(2,1,2)
sns.boxplot(hd[hd["target"]==1]["trestbps"])
plt.xlim(90, 210)
plt.title("has disease")
plt.subplots_adjust(hspace=0.4)




fig = plt.figure(figsize=(16,8))
We wouldsns.heatmap(hd.corr().sort_values("target", ascending=False)[["target"]], annot=True) #select only the target feature and sort the correlation




pd.get_dummies(hd["cp"], prefix="cp")
pd.get_dummies(hd["thal"], prefix="thal")
pd.get_dummies(hd["slope"], prefix="slope")
hd = pd.concat([hd, pd.get_dummies(hd["cp"], prefix="cp"), pd.get_dummies(hd["thal"], prefix="thal"), pd.get_dummies(hd["slope"], prefix="slope")], axis=1)
hd.drop(["cp", "thal", "slope"], axis=1, inplace=True)
hd




x = hd.drop(["target", "age_groups"], axis=1)
y = hd["target"]
scalar = StandardScaler()

lr = LogisticRegression()

pipeline = Pipeline([('transformer', scalar), ('estimator', lr)])

#lr.fit(x_train,y_train)
#acc = lr.score(x_test,y_test)*100
acc = cross_val_score(pipeline, x, y, cv=5)

print(acc)
print("Accuracy: {:.3f} (+/- {:.3f})".format(acc.mean(), acc.std()*2))




scoreList = []
for i in range(2,31):
    knn = KNeighborsClassifier(n_neighbors = i)
    cv = StratifiedKFold(n_splits=5)
    pipeline = Pipeline([('transformer', scalar), ('estimator', knn)])

    acc = cross_val_score(pipeline, x, y, cv=cv, scoring="f1")
    scoreList.append(acc.mean())
    #print(acc)
    print("Accuracy: {:.3f} (+/- {:.3f})  Neighbors: {}".format(acc.mean(), acc.std()*2,i))

print("\nMax accuracy achieved: {:.3f}".format(max(scoreList)))
plt.figure(figsize=(16,8))
plt.plot(scoreList)




rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
cv = StratifiedKFold(n_splits=7)
pipeline = Pipeline([('transformer', scalar), ('estimator', rf)])

acc = cross_val_score(pipeline, x, y, cv=cv, scoring="f1")
scoreList.append(acc.mean())
#print(acc)
print("Accuracy: {:.3f} (+/- {:.3f})".format(acc.mean(), acc.std()*2))




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
rf.fit(x_train, y_train)

fi = pd.DataFrame(rf.feature_importances_, index = x.columns, columns=['importance']).sort_values('importance', ascending=False)
plt.figure(figsize=(20,8))
sns.heatmap(fi, annot=True, cmap=sns.diverging_palette(10, 140, s=90, l=60, as_cmap=True))

