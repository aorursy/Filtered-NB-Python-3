#!/usr/bin/env python
# coding: utf-8



import pandas as pd
pd.options.display.max_colwidth = 200
pd.options.display.max_columns = 200
import numpy as np

import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict,cross_val_score,train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve

import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




df = pd.read_csv('../input/creditcard.csv')




df['hour'] = df['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)




df.pivot_table(values='Amount',index='hour',columns='Class',aggfunc='count')




def PlotHistogram(df,norm):
    bins = np.arange(df['hour'].min(),df['hour'].max()+2)
    plt.figure(figsize=(15,4))
    sns.distplot(df[df['Class']==0.0]['hour'],
                 norm_hist=norm,
                 bins=bins,
                 kde=False,
                 color='b',
                 hist_kws={'alpha':.5},
                 label='Legit')
    sns.distplot(df[df['Class']==1.0]['hour'],
                 norm_hist=norm,
                 bins=bins,
                 kde=False,
                 color='r',
                 label='Fraud',
                 hist_kws={'alpha':.5})
    plt.xticks(range(0,24))
    plt.legend()
    plt.show()




start = time.time()
print('Normalized histogram of Legit/Fraud over hour of the day')
PlotHistogram(df,True)
print('Counts histogram of Legit/Fraud over hour of the day')
print('*you can barely see the Fraud cases since there are so little of them.')
PlotHistogram(df,False)
print(time.time()-start)




print('Fraud is {}% of our data.'.format(df['Class'].value_counts()[1] / float(df['Class'].value_counts()[0])*100))




mask_true = (df['Class'] == 1.0) 
mask_false = (df['Class'] == 0.0)

df['Amount'] = StandardScaler().fit_transform(df[['Amount']])




def PlotViolins(minHour,maxHour):
    plt.figure(figsize=(15,6))
    plt.title('Amount by class throughout the day')
    plt.ylim([-1,3.0])
    sns.violinplot(data=df[df['hour'].isin(range(minHour,maxHour+1))],x='hour',y='Amount',hue='Class',split=True,palette='Set2',cut=0)
    plt.legend(loc='lower right')
    plt.show()
PlotViolins(0,11)
PlotViolins(12,23)




# Model building
# Let's start with a vanilla Logistic Regression since it seems like for some of the features, a sigmoid curve can sort of separate the classes.sns.pairplot(data=pd.concat([df.loc[:,'hour'],df.loc[:,'V1':'V6'],df.loc[:,'Class']],axis=1),
#              hue='Class',
#              diag_kind='kde',
#              plot_kws={'alpha':0.2})




features = pd.concat([df.loc[:,'V1':'Amount'],df.loc[:,'Time']],axis=1)
target = df['Class']

x_train,x_test,y_train,y_test = train_test_split(features,target, stratify=target,test_size=0.35, random_state=1)

print('y_train class counts')
print(y_train.value_counts())
print('')
print('y_test class counts')
print(y_test.value_counts())


# Let's store our y_test legit and fraud counts for normalization purposes later on
y_test_legit = y_test.value_counts()[0]
y_test_fraud = y_test.value_counts()[1]




lr_model = LogisticRegression()
lr_model.fit(x_train,y_train)

pred = lr_model.predict(x_test)




def PlotConfusionMatrix(y_test,pred,y_test_legit,y_test_fraud):

    cfn_matrix = confusion_matrix(y_test,pred)
    cfn_norm_matrix = np.array([[1.0 / y_test_legit,1.0/y_test_legit],[1.0/y_test_fraud,1.0/y_test_fraud]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(norm_cfn_matrix,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)

    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    plt.show()
    
    print('---Classification Report---')
    print(classification_report(y_test,pred))

PlotConfusionMatrix(y_test,pred,y_test_legit,y_test_fraud)




lr_model = LogisticRegression(class_weight='balanced')
lr_model.fit(x_train,y_train)

pred = lr_model.predict(x_test)

PlotConfusionMatrix(y_test,pred,y_test_legit,y_test_fraud)




for w in [1,5,10,100,500,1000]:
    print('---Weight of {} for Fraud class---'.format(w))
    lr_model = LogisticRegression(class_weight={0:1,1:w})
    lr_model.fit(x_train,y_train)

    pred = lr_model.predict(x_test)
    PlotConfusionMatrix(y_test,pred,y_test_legit,y_test_fraud)




fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(1,2,1)
ax1.set_xlim([-0.05,1.05])
ax1.set_ylim([-0.05,1.05])
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve')

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlim([-0.05,1.05])
ax2.set_ylim([-0.05,1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')

for w,k in zip([1,5,10,20,50,100,10000],'bgrcmykw'):
    lr_model = LogisticRegression(class_weight={0:1,1:w})
    lr_model.fit(x_train,y_train)
    pred_prob = lr_model.predict_proba(x_test)[:,1]

    p,r,_ = precision_recall_curve(y_test,pred_prob)
    tpr,fpr,_ = roc_curve(y_test,pred_prob)
    
    ax1.plot(r,p,c=k,label=w)
    ax2.plot(tpr,fpr,c=k,label=w)
ax1.legend(loc='lower left')    
ax2.legend(loc='lower left')

plt.show()






