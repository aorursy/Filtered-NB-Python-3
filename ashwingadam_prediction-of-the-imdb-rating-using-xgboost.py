#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import IPython.display as dispaly
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import xgboost as xgb
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from __future__ import division
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




full_data=pd.read_csv('../input/movie_metadata.csv')




list(full_data)




full_data.describe()




full_data.info()




plt.plot(full_data['budget'])




numerical_var=[x for x in full_data.columns if 
               full_data.dtypes[x] != 'object']




catagorical_var=[x for x in full_data.columns if x not in numerical_var]
catagorical_var




sns.distplot(full_data['imdb_score']**3)
plt.show()




(full_data['imdb_score']**2).skew()




print ((full_data).kurt())
print ('--------------------')
print ((full_data).skew())




print (pd.isnull(full_data[numerical_var]).sum())
print ('--------------------------')
print (pd.isnull(full_data[catagorical_var]).sum())




full_data['gross']=full_data['gross'].fillna(0)
full_data['budget']=full_data['budget'].fillna(0)
full_data['title_year']=full_data['title_year'].fillna(0)
full_data['aspect_ratio']=full_data['aspect_ratio']    .fillna(full_data['aspect_ratio'].mean())
full_data['content_rating']=full_data['content_rating'].fillna('R')
full_data['color']=full_data['color'].fillna('Color')
full_data[numerical_var]=full_data[numerical_var]    .fillna(full_data[numerical_var].mean())




full_data=full_data.dropna()
full_data.shape




full_data['actor_1_movie_fb_likes']=(full_data['actor_1_facebook_likes']        *full_data['movie_facebook_likes']).astype(np.float64)
full_data['actor_2_movie_fb_likes']=(full_data['actor_2_facebook_likes']        *full_data['movie_facebook_likes']).astype(np.float64)
full_data['actor_3_movie_fb_likes']=(full_data['actor_3_facebook_likes']        *full_data['movie_facebook_likes']).astype(np.float64)
full_data['director_name_fb_likes']=(full_data['director_facebook_likes']        *full_data['movie_facebook_likes']).astype(np.float64)




full_data['actor_1_name_imdb_score']=(full_data[['actor_1_name','imdb_score']].groupby('actor_1_name').mean()).astype(np.float64)
full_data['actor_2_name_imdb_score']=(full_data[['actor_2_name','imdb_score']].groupby('actor_2_name').mean()).astype(np.float64)
full_data['actor_3_name_imdb_score']=(full_data[['actor_3_name','imdb_score']].groupby('actor_3_name').mean()).astype(np.float64)
full_data['director_name_imdb_score']=(full_data[['director_name','imdb_score']].groupby('director_name').mean()).astype(np.float64)




updating_coloumns=['actor_1_name_imdb_score','actor_2_name_imdb_score',
                   'actor_3_name_imdb_score','director_name_imdb_score']
refered_coloumns=['actor_1_name','actor_2_name','actor_3_name','director_name']
k=0
for j in updating_coloumns:
    temp=refered_coloumns[k]
    a=(full_data[[refered_coloumns[k],'imdb_score']].groupby(refered_coloumns[k]).mean()).astype(np.float64)
    index_mapping={}
    for i in list(a.index):
        index_mapping[i]= a.loc[i,'imdb_score']
    full_data[j]=full_data[refered_coloumns[k]].map(index_mapping)
    k=k+1




print (full_data.shape)




dropping_var=['director_name','actor_2_name','actor_1_name'
    ,'movie_title','actor_3_name','plot_keywords','movie_imdb_link' ]
full_data=full_data.drop(dropping_var,axis=1)
print (full_data.shape)




numerical_var=[x for x in full_data.columns if 
               full_data.dtypes[x] != 'object']
catagorical_var=[x for x in full_data.columns if x not in numerical_var]




print (catagorical_var)
print ('-----------------')
print (numerical_var)




full_data=pd.get_dummies(full_data)




skewed=full_data[numerical_var].apply(lambda x:stats.skew(x.dropna()))
skewed=skewed[skewed>0.75]
skewed1=skewed[skewed<-0.75]
print (skewed1)
skewed=skewed.index
skewed1=skewed1.index
full_data[skewed1]=full_data[skewed1]**3
full_data[skewed]=np.log1p(full_data[skewed])




full_data.shape




x=full_data
y=full_data['imdb_score']
x=x.drop(['imdb_score'],axis=1)




x_train,x_test,y_train,y_test=    train_test_split(x,y,test_size=0.3,random_state=0)




print (x_train.shape,y_train.shape)
print ('---------------')
print (x_test.shape,y_test.shape)




clf=RandomForestRegressor(n_estimators=500)
clf.fit(x_train,y_train)




feature_imp_mapping={}
for features,importance in zip(full_data.columns, clf.feature_importances_):
    if importance!=0:
        feature_imp_mapping[features]=importance
    

importances = pd.DataFrame.from_dict(feature_imp_mapping, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance')
importances.index




plt.figure(figsize=(10,200))
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(importances.index)), importances['Gini-importance'], color='b', align='center')
plt.yticks(range(len(importances.index)), importances.index)
plt.xlabel('Relative Importance')
#plt.savefig('/home/ashwin/Downloads/imp.png')
plt.show()




x_train=x_train[['actor_2_name_imdb_score',   
'actor_3_movie_fb_likes',
'director_name_fb_likes',
'gross',
'duration',
'budget',
'actor_1_name_imdb_score',
'title_year',
'num_user_for_reviews',
'actor_3_name_imdb_score',
'actor_2_facebook_likes',
'director_facebook_likes',
'num_critic_for_reviews',
'num_voted_users']]
x_test=x_test[['actor_2_name_imdb_score',   
'actor_3_movie_fb_likes',
'director_name_fb_likes',
'gross',
'duration',
'budget',
'actor_1_name_imdb_score',
'title_year',
'num_user_for_reviews',
'actor_3_name_imdb_score',
'actor_2_facebook_likes',
'director_facebook_likes',
'num_critic_for_reviews',
'num_voted_users']]




gbm = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
colsample_bytree=1, max_depth=7).fit(x_train, y_train)
predictions = gbm.predict(x_test)




print(explained_variance_score(predictions,y_test))




d={'predictions':predictions,
   'actual':y_test}
data_predicted_actual=pd.DataFrame(d)




data_predicted_actual

