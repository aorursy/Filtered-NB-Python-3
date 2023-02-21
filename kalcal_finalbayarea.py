#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import the 'bay_area' sheet
data=pd.read_csv("../input/bay_area.csv")
print(data.head(10))
print(data.info())




#Dropping columns 0,3,15,17,18
data= data.drop(data.columns[[0,3,15,17,18]],axis=1)
print(data.info())




#Look at info column
print(data.iloc[0,1])
#all data present in info column is available as other columns apart from lot size, so we keep just that
data.rename(columns={'info': 'lotsize'}, inplace=True)
datatemp = pd.DataFrame(data.lotsize.str.split('Lot size: ',1).tolist(), columns = ['drop','keep'])
print(datatemp.head())
#copy the keep col to lotsize in data 
data['lotsize']=datatemp.keep.astype(int)
print(data.head())




#graph of each column to have an idea of the spread of the data
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,20))
print(plt.show())




#address and 'latitude and longtitude' give the same information, so that can be dropped
data= data.drop(data.columns[0],axis=1)

#count of unique values of last sold date
print(data.groupby('lastsolddate')['lastsolddate'].count().sort_values())
print(data.info())




#Change lastdoldate column to a date column (right know it is an object)
data['lastsolddate'] =  pd.to_datetime(data['lastsolddate'])
print(data.info())
print(data.lastsolddate.min())
print(data.lastsolddate.max())
#The lastsolddate data is between Oct 1970 and March 2016




#Neighborhood
#number of unique types in neighborhood column to see if we have to keep
#we have 71 types of neighborhoods
print(data.groupby('neighborhood')['neighborhood'].count().sort_values())
print(data.groupby('neighborhood')['lastsoldprice'].mean().sort_values())




#usecode
print(data.groupby('usecode')['usecode'].count().sort_values())
print(data.groupby('usecode')['lastsoldprice'].mean().sort_values()) #CLear correlation seen

#We can map these to numbers
data['usecode']=data['usecode'].map({'MultiFamily5Plus':1,'Townhouse':2,
                                     'Cooperative':3,'Condominium':4,
                                     'MultiFamily2To4':5,'Apartment':6,
                                     'SingleFamily':7,'Duplex':8,'Miscellaneous':9,
                                     'Mobile':10}).astype(int)
print(data.head())




#Zindex value
data['zindexvalue'] = data['zindexvalue'].str.replace(',', '')
data['zindexvalue']=data['zindexvalue'].astype(int)

#yearbuilt
print(data.yearbuilt.min())
print(data.yearbuilt.max())
#House ranges from 1860 to 2016
#We can think of using suitable bands to see how prices are differenct
data.plot(kind='scatter', x='yearbuilt', y='lastsoldprice')

#remove outliers
print(data.lastsoldprice.quantile([0.02,0.98]))

#adding cap and floor
data.loc[(data.lastsoldprice > 4100000.0) ,'lastsoldprice'] =4100000.0
data.loc[(data.lastsoldprice < 320000.0) ,'lastsoldprice'] =320000.0
data['lastsoldprice'] = data['lastsoldprice'].astype(float)

#check to see if cap implemented
a=data[['lastsoldprice']]>4100000
print(a.groupby('lastsoldprice')['lastsoldprice'].count())

data.plot(kind='scatter', x='yearbuilt', y='lastsoldprice') #outliers removal can be seen




#final check
print(data.head())




#Check correlation for all these columns with lastsoldprice
print(data.corr())
#just get the corr with lastsoldprice
print(data.corr()["lastsoldprice"].sort_values(ascending=False))




#use longtitude and latitude to see how prices vary
import matplotlib.pyplot as plt
data.plot(kind='scatter', x='longitude', y='latitude', s=(data.lastsoldprice/1000000))
print(plt.show())
#The last sold price above is divided by 1M so that the graph makes more sense, else the graph will just get covered up in terms of size due to the big numbers of the house prices
#It is seen from the data that there are areas in Bay_area where the house prices are relatively higher




#Approach of latitiude and longtitude does not work right on their own as two columns
#we need to find a way to address them together as a location, rather than just treating them as two different var when predicting lastsoldprice
#We can use KNN Neighbours to cluster them into different regions

#MODEL
#for machine learning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

#KMeans= used for unsupervised clustering 
a=data[['latitude','longitude']]
from sklearn.cluster import KMeans
model=KMeans(n_clusters=5)
model.fit(a)
labels=model.predict(a) #labels has the cluster values assigned for each column
#Plotting the above color coded for the clusters
XS=a.iloc[:,0]
YS=a.iloc[:,1]
plt.scatter(XS,YS,c=labels)

centroids=model.cluster_centers_
cent_x=centroids[:,0]
cent_y=centroids[:,1]
plt.scatter(cent_x,cent_y,marker='D',s=50)

print(plt.show())

#Evaluating my cluster number (6) INERTIA
print(model.inertia_)




#See what the inertia values are like for different number of clusters- 3-7
from sklearn.cluster import KMeans
ks=range(3,15)
iner=[]
for k in ks:
    model=KMeans(n_clusters=k)
    model.fit(a)
    iner.append(model.inertia_)

plt.plot(ks,iner,'-o')
plt.xticks(ks)
plt.show()

#A low inertia is good, in the graph we try choose the cluster where intertia begins to decrease more slowly
#Since I do not want too many groups in the region, I will start with 6 clusters(regions)




#Going to add the cluster label column to the main data DataFrame
print(labels)
data['LatLongCluster']=labels
data.info()
#Neighborhood
#number of unique types in neighborhood column to see if we have to keep
#we have 71 types of neighborhoods
clus=range(0,5)
uni=[]
for c in clus:
    a=data[data['LatLongCluster'] == c]
    uni.append(a.neighborhood.nunique())
    
print(uni)     #the count more than 71 
#(25 overlaps, since this the best case we going to take this to be a close approximate for 
#neighbour col, and hence drop it)

#But before that we are going to calculate the price per square feet and see if there is correlation between the neighborhood and pricepersquarefeet
data['PricePerFeet']=data['lastsoldprice']/data['finishedsqft']
print(data.head(1))

#Create a new dataframe called temp
#Price per feet of each cluster
avgprice=(data.groupby('LatLongCluster')['PricePerFeet'].mean())  #Cluster 3 seems to be very expensive

#number of houses under each cluster
houses=(data.groupby('LatLongCluster')['LatLongCluster'].count())

temp=pd.concat([houses,avgprice],axis=1)
temp.columns = ['houses', 'priceperfeet']
print(temp)




#Look at the data now
print(data.info())

#drop unecessary columns
data.drop(data.columns[[4,6,7,8,14]], axis=1, inplace=True)
print(data.head())




#Get correlation of these columns with lastsoldprice
print(data.corr()["lastsoldprice"].sort_values(ascending=False))




#cross correlation to remove any column (last check)
print(data.corr())




#split data
X=data.drop(['lastsoldprice'],axis=1) #all indepdendant variables

Y=data['lastsoldprice'] #target variables

#splitting data as test and train, testing on 20% of the data and training on the 80.
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)




#First model we going to try is Linear Regression
from sklearn import linear_model
reg_all=linear_model.LinearRegression()

reg_all.fit(X_train,Y_train) #fitting the model for the x and y train

y_pred=reg_all.predict(X_test) #predicting y(the target variable), on x test

Rsquare=reg_all.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))

#List all the indepedent variables with the coeffecient from the linear regression model
coeff_df = pd.DataFrame(X_train.columns.delete(0))
coeff_df.columns = ['Variable']
coeff_df["Coeff"] = pd.Series(reg_all.coef_)

coeff_df.sort_values(by='Coeff', ascending=True)
print(coeff_df)

#print intercept and rmse
print("Intercept: %f" %(reg_all.intercept_))
rmse=np.sqrt(mean_squared_error(Y_test,y_pred))
print("rmse: %f" %(rmse))

#cross validation, to check the different R2 values we get with different combinations of test and train (same 20:80), five folds done here
cv_results=cross_val_score(reg_all,X,Y,cv=5)
mean=np.mean(cv_results)

print("CV results mean: %f" %(mean))
print("List of R2 from five fold CV:")
print(cv_results)




#Second model we going to try is Log Reg
from sklearn.linear_model import LogisticRegression
reg_all=linear_model.LogisticRegression()

reg_all.fit(X_train,Y_train) #fitting the model for the x and y train

y_pred=reg_all.predict(X_test) #predicting y(the target variable), on x test

Rsquare=reg_all.score(X_test,Y_test)
print("Rsquare: %f" %(Rsquare))

#Equation coefficient and Intercept
print('Coefficient: \n', reg_all.coef_)
print('Intercept: \n', reg_all.intercept_)
#Predict Output
predicted= model.predict(x_test)




#seeing few
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#split data
X=data.drop(['lastsoldprice'],axis=1) #all indepdendant variables
Y=data['lastsoldprice'] #target variables

#splitting data as test and train, testing on 20% of the data and training on the 80.
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

models = [LinearRegression(),
              RandomForestRegressor(n_estimators=100, max_features='sqrt'),
              KNeighborsRegressor(n_neighbors=6),
              SVR(kernel='linear'),
              LogisticRegression()
              ]
 
TestModels = pd.DataFrame()
tmp = {}

for model in models:
        # get model name
        m = str(model)
        tmp['Model'] = m[:m.index('(')]
        # fit model on training dataset
        model.fit(X_train, Y_train)
        # predict prices for test dataset and calculate r^2
        tmp['R2_Price'] = %(Rsquare)
        # write obtained data
        TestModels = TestModels.append([tmp])
        
TestModels.set_index('Model', inplace=True)
 
fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
print(plt.show())






