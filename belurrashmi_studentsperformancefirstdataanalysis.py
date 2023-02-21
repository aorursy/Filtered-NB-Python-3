#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




data=pd.read_csv("../input/StudentsPerformance.csv")




data.head()




data = data.rename(
    columns={'race/ethnicity': 'race_ethnicity', 
             'parental level of education': 'parental_education',
             'test preparation course':'test_prep',
              'math score': 'math',
             'reading score':'reading',
             'writing score' :'writing'})




data.head()




data['language']=(data.reading+data.writing)/2




dataT=pd.melt(data,id_vars=['gender', 'race_ethnicity', 'parental_education', 'lunch', 'test_prep'],value_vars=['math', 'reading', 'writing'])




dataT = dataT.rename(
    columns={'variable':'subject',
             'value' :'score'})




data.head()




sn.lmplot(x='math',y='reading',data=data,hue='gender')




sn.lmplot(x='math',y='writing',data=data,hue='gender')




sn.lmplot(x='reading',y='writing',data=data,hue='gender')




sn.lmplot(x='math',y='language',data=data,hue='gender')




sn.lmplot(x='math',y='language',data=data,hue='race_ethnicity')




sn.lmplot(x='math',y='language',data=data,hue='parental_education')




sn.lmplot(x='math',y='language',data=data,hue='lunch')
#lunch difinetely shows the difference




sn.lmplot(x='math',y='language',data=data,hue='test_prep')
#test_prep difinetely shows the difference




data["genderF"]=np.where(data['gender']=='female',1,0)




data.race_ethnicity.value_counts()




data['RaceA']=np.where(data['race_ethnicity']=='group A',1,0)
data['RaceB']=np.where(data['race_ethnicity']=='group B',1,0)
data['RaceC']=np.where(data['race_ethnicity']=='group C',1,0)
data['RaceD']=np.where(data['race_ethnicity']=='group D',1,0)
data['RaceE']=np.where(data['race_ethnicity']=='group E',1,0)
data['RaceABC']=np.where((data['race_ethnicity']=='group A')|(data['race_ethnicity']=='group B')|(data['race_ethnicity']=='group C'),1,0)
data['RaceDE']=np.where((data['race_ethnicity']=='group E')|(data['race_ethnicity']=='group D'),1,0)




data.parental_education.value_counts()




data['Parent_highSchool']=np.where((data['parental_education']=='high school') | (data['parental_education']=='some high school'),1,0 )
data['Parent_College']=np.where(data['parental_education']=='some college' ,1,0 )
data['Parent_associate']=np.where(data['parental_education']=="associate's degree" ,1,0 )
data['Parent_bachelor']=np.where(data['parental_education']=="bachelor's degree" ,1,0 )
data['Parent_masters']=np.where(data['parental_education']=="master's degree" ,1,0 )
data['Parent_higherthanHighschool']=np.where((data['parental_education']=='high school') | (data['parental_education']=='some high school'),0,1 )




data['lunchF']=np.where(data['lunch']=='free/reduced' ,1,0 )
data['test_prepDone']=np.where(data['test_prep']=='completed' ,1,0 )




data.head()




dataPreped=data[['math', 'reading', 'writing', 'language', 'genderF','RaceABC', 'Parent_higherthanHighschool',  'lunchF',
       'test_prepDone']]




dataPreped.head()





corr=dataPreped.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})




Xlang=data[[ 'math', 'genderF','RaceABC', 'Parent_higherthanHighschool',  'lunchF',
       'test_prepDone']]
ylang=data['language']
Xmath=data[[ 'language', 'genderF','RaceABC', 'Parent_higherthanHighschool',  'lunchF',
       'test_prepDone']]
ymath=data['math']
X=data[[  'genderF','RaceABC', 'Parent_higherthanHighschool',  'lunchF',
       'test_prepDone']]




from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix
from sklearn import preprocessing
from sklearn import utils
from sklearn import ensemble




def LinearModel(Xlang, ylang ):
    Xlang_train, Xlang_test, ylang_train, ylang_test = train_test_split(Xlang, ylang, test_size=0.33, random_state=42)
    #Linear Regression for language
    LinReg = LinearRegression()
    reg=LinReg.fit(Xlang_train, ylang_train)
    reg.score(Xlang_train, ylang_train) #R square value
    ylangPredict=reg.predict(Xlang_test)
    #return coefficinents
    print("Co-efficients")
    print (reg.coef_)
    print("Mean squared error: %.2f"
      % mean_squared_error(ylang_test, ylangPredict))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ylang_test, ylangPredict))
    return (ylangPredict)




LangPrediction =LinearModel(Xlang, ylang )




mathPrediction=LinearModel(Xmath,ymath)




LangPrediction =LinearModel(X, ylang )




mathPrediction1=LinearModel(X,ymath)




def LogisticModel(Xlang, ylang ):
    Xlang_train, Xlang_test, ylang_train, ylang_test = train_test_split(Xlang, ylang, test_size=0.33, random_state=42)
    #Logistic Regression for language
    Reg = LogisticRegression()
    reg=Reg.fit(Xlang_train, ylang_train)
    reg.score(Xlang_train, ylang_train) #R square value
    ylangPredict=reg.predict(Xlang_test)
    #return coefficinents
    print("Co-efficients")
    print (reg.coef_)
    print("Mean squared error: %.2f"
      % mean_squared_error(ylang_test, ylangPredict))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(ylang_test, ylangPredict))
    return (ylangPredict)




def GradBoosting(Xlang, ylang):

# #############################################################################
# Load data
    Xlang_train, Xlang_test, ylang_train, ylang_test = train_test_split(Xlang, ylang, test_size=0.33, random_state=42)
# #############################################################################
# Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(Xlang_train, ylang_train)
    mse = mean_squared_error(ylang_test, clf.predict(Xlang_test))
    print("MSE: %.4f" % mse)

# #############################################################################
# Plot training deviance

# compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(Xlang_test)):
        test_score[i] = clf.loss_(ylang_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',  label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
# Plot feature importance
    feature_importance = clf.feature_importances_
# make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, Xlang.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()




GradBoosting(Xlang,ylang)




GradBoosting(Xmath,ymath)




GradBoosting(X,ylang)




GradBoosting(X,ymath)

