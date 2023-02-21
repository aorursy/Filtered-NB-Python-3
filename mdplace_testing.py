#!/usr/bin/env python
# coding: utf-8



print ("testing to see if this works as expected")




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns




df_raw=pd.read_csv("../input/train.csv")

#df_raw=pd.read_csv("train.csv")




df_raw.head()
df_clean = df_raw.set_index('PassengerId')




df_clean




def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()



# Plot distributions of Age of passangers who survived or did not survive
plot_distribution( df_clean , var = 'Age' , target = 'Survived' , row = 'Sex' )




df_clean.Age.unique()




df_clean.Age.mean()




## Replacing nans with the mean age
df_clean.Age.replace(to_replace='nan',value=df_clean.Age.mean(),inplace=True)




## Getting dummies for p class
pclass_dums = pd.get_dummies(df_clean.Pclass)

## Getting dummies for Port Embarked
embark_dums = pd.get_dummies(df_clean.Embarked)

## Getting dummies for age quartiles and renaming them
age_dums = pd.get_dummies(pd.qcut(df_clean.Age,q=5))
age_dums.columns=['AgeBand_' + str(x) for x in range(1,6)]









## Function to get dummies for Male
def male(x):
   if x == 'male':
       return(1)
   else:
       return(0)    
   
df_clean["Male"]= df_clean.Sex.apply(male)
  




df_full_features =df_clean.join(age_dums).join(embark_dums).join(pclass_dums)
df_full_features.head(2)




##Pulling Out Column Values
features =df_full_features.columns.values.tolist()
print(features)




##Pulling Out Column Values
df_features_1 = df_full_features[['Male', 'AgeBand_1', 'AgeBand_2',
                                  'AgeBand_3', 'AgeBand_4', 
                                   'C', 'Q', 1, 2]]


df_features_1.head()




## Importing models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score




## Preppping model
lm = LogisticRegression()




y = df_clean.Survived
x = df_features_1
lm.fit(x,y)









#Getting Overall predicitive power of the model
LM_Score = pd.DataFrame()
LM_Score['Metric']=['R2','AUC']
LM_Score['Scores']=[lm.score(x,y),roc_auc_score(y,lm.predict(x))]

LM_Score









#Getting the Coefficients and Odds for each features
feature_names =x.columns.values.tolist()
LM_Coef = pd.DataFrame()
LM_Coef['Features']= feature_names
LM_Coef['Coefficients']=lm.coef_[0]
LM_Coef['Odds']=np.exp(lm.coef_[0])
LM_Coef.sort_values(by='Coefficients',ascending=False)




actuals = lm.predict(x) 
probas = lm.predict_proba(x)       
plt.plot(roc_curve(y, probas[:,1])[0], roc_curve(y, probas[:,1])[1])




roc_auc_score(y,lm.predict(x))


























