# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas
os.chdir("../input/new-york-city-airbnb-open-data")
df=pandas.read_csv("AB_NYC_2019.csv")

df.head()

import seaborn as sns

df.describe()

df.isna().sum()

df.fillna(value={"reviews_per_month":0},inplace=True)

df.isna().sum()


df1=df.drop(columns=["id","name","host_id","host_name","last_review","neighbourhood"])
df2=df.drop(columns=["id","host_id","host_name","last_review"])

df1.isna().sum()

df2.isna().sum()



df1.nunique()

import seaborn as sns
from matplotlib import pyplot as plt

sns.heatmap(df1.corr())

g = sns.PairGrid(df1)
g.map(plt.scatter)

sns.catplot(x="neighbourhood_group",y="price",data=df1)

sns.catplot(x="room_type",y="price",data=df1)

X1=df1.loc[:,df1.columns!="price"]
X1.head()

Y=df1["price"]
Y

X1=pd.get_dummies(data=X1,columns=["neighbourhood_group","room_type"],drop_first=False)

X1.head()

 from sklearn.model_selection import train_test_split as tts


X_train, X_test, y_train, y_test = tts(X1,Y, test_size=0.2, random_state=42)

from sklearn.preprocessing import scale
X_train=scale(X_train)
X_test=scale(X_test)

from xgboost import XGBRegressor as xgbr

xgb=xgbr(n_estimators=60,subsample=0.7,colsample_bytree=0.2,max_depth=2,gamma=0,random_state=42)

xgb2=xgbr(n_estimators=1000, learning_rate=0.1, early_stopping=5, max_depth=5, min_child_weight=1 )

xgb2.fit(X_train,y_train)


xgb.fit(X_train,y_train)

from sklearn.metrics import r2_score as rs

#pred2=xgb2.predict(X_test)
pred1=xgb.predict(X_test)

rs(y_test,pred1)

rs(y_test,pred2)

from catboost import CatBoostRegressor

model = CatBoostRegressor(
    n_estimators = 2500,
    loss_function = 'MAE',
    eval_metric = 'RMSE',
    cat_features = ["neighbourhood","neighbourhood_group","room_type"])

xt,xe,yt,ye=tts(X_train,y_train,test_size=0.05,random_state=42)

os.chdir("/kaggle/working/")

model.fit( xt, yt, use_best_model=True, eval_set=(xe, ye), silent=True, plot=True )



predi=model.predict(X_test)

pp=(pred1+predi)/2

rs(y_test,pp)

landmarks={"CP":[40.785091, -73.968285],"SL":[40.6892,-74.0445],"ESB":[40.7484,-73.9857],"TS":[40.7580, -73.9855],"MMA":[40.7614, -73.9776],"BB":[40.7061,-73.9969],"RC":[40.7587 , -73.9787],"HL":[40.7480, -74.0048]}



locX=df1["latitude"].to_list()
locY=df1["longitude"].to_list()


locloc=[0]*len(locX)
for key,value in landmarks.items():
    for i in range(len(locX)):
        locloc[i]+=((value[0]-locX[i])**2+(value[1]-locY[i])**2)**0.5
for i in locloc:
    i/=8

locloc

df1["avgdist"]=locloc

df1.head(6)


