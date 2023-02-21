#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
#import pandas_profiling
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

#%matplotlib inline




df_train = pd.read_csv('../input/train.csv')
df_predict = pd.read_csv('../input///.test.csv')




#df_train.head().T




df_train.shape









# Generate a profile report to examine
pandas_profiling.ProfileReport(df_train)




def data_preprocessing(df):
    # Drop column which have very high number of zeros
    df.drop(['num_private'], axis=1, inplace=True)
    
    # Impute Zeros
    
    # Drop column scheme_name because it has 47.5% missing values
    df.drop(['scheme_name'], axis=1, inplace=True)
    
    # Impute missing values
    # All missing values of latitude, longitude and height (listed as 0s) 
    # are converted to the mean value within their administrative region
    a= df[df["longitude"] < 1]
    a.iloc[:,df.columns == "latitude"]= np.nan
    a.iloc[:,df.columns == "longitude"]= np.nan
    df[df["longitude"] < 1] = a
    df["longitude"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).longitude
    df["latitude"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).latitude

    a= df[df["gps_height"] < 1]
    a.iloc[:,df.columns == "gps_height"]= np.nan
    df[df["gps_height"] < 1] = a
    df["gps_height"] = df.groupby("region_code").transform(lambda x: x.fillna(x.mean())).gps_height

    df=df.fillna(df.mean())
    df=df.fillna('Unknown')

    # Drop the column having 1 constant value
    df.drop(['recorded_by'], axis=1, inplace=True)
    
    # Drop the Duplicate columns
    df.drop(['payment'], axis=1, inplace=True)
    df.drop(['quantity_group'], axis=1, inplace=True)
    df.drop(['extraction_type'], axis=1, inplace=True)
    df.drop(['source'], axis=1, inplace=True)
    df.drop(['waterpoint_type_group'], axis=1, inplace=True)
    df.drop(['quality_group'], axis=1, inplace=True)
    df.drop(['management'], axis=1, inplace=True)
    
    # Add column operational_age
    df.construction_year=pd.to_numeric(df.construction_year)
    df.loc[df.construction_year <= 0, df.columns=='construction_year'] = 2001
    df['operational_age']=df.date_recorded.apply(pd.to_datetime)-     df.construction_year.apply(lambda x: pd.to_datetime(x,format='%Y'))
    df.operational_age=df.operational_age.astype('timedelta64[D]').astype(int)
    
    # Add column days_since_recorded, idea being more recently recorded pumps might be more likely to be functional. 
    df.date_recorded = pd.to_datetime(df.date_recorded)
    # The most recent data is 2013-12-03. Subtract each date from this point to get 'days_since_recorded' column.
    df['days_since_recorded'] = pd.datetime(2013, 12, 3) - pd.to_datetime(df.date_recorded)
    df.days_since_recorded = df.days_since_recorded.astype('timedelta64[D]').astype(int)

    # Change construction year into decade buckets to reduce the number of Dummy columns. 
    def construction_year_wrangle(row):
        if row['construction_year'] >= 1960 and row['construction_year'] < 1970:
            return '1960s'
        elif row['construction_year'] >= 1970 and row['construction_year'] < 1980:
            return '1970s'
        elif row['construction_year'] >= 1980 and row['construction_year'] < 1990:
            return '1980s'
        elif row['construction_year'] >= 1990 and row['construction_year'] < 2000:
            return '1990s'
        elif row['construction_year'] >= 2000 and row['construction_year'] < 2010:
            return '2000s'
        elif row['construction_year'] >= 2010:
            return '2010s'
        else:
            return 'unknown'
    df['construction_year'] = df.apply(lambda row: construction_year_wrangle(row), axis=1)

    # Categorical variables to dummy variables. 
    cols = ['scheme_management','construction_year','extraction_type_group','extraction_type_class',
                  'management_group','payment_type','water_quality','quantity','source_type',
                  'source_class','waterpoint_type']  
    df = pd.get_dummies(df, columns = cols)
    
    # Categorical variables change Names to numbers 
    columns=['funder','installer','basin','region','lga','public_meeting','permit']
    for var in columns:
        df[var] = preprocessing.LabelEncoder().fit_transform(df[var])
    
    # too many distinct values with top 5 values not dominating hence dropping
    df.drop(['wpt_name'], axis=1, inplace=True)
    df.drop(['subvillage'], axis=1, inplace=True)
    df.drop(['ward'], axis=1, inplace=True)
    df.drop(['permit'], axis=1, inplace=True)

    # Drop not usefull columns
    df.drop(['date_recorded'], axis=1, inplace=True)

    return df




df_train = data_preprocessing(df_train)
df_predict = data_preprocessing(df_predict)




#df_train.to_csv('./train_clean.csv', index=False)
#df_predict.to_csv('./predict_clean.csv', index=False)




#y = df_train['defective'].values
y = df_train['defective']
X = df_train.drop(['defective', 'new_ids'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)




rfc_params = {'n_estimators':[300,400,500,600],
             'criterion':['gini', 'entropy'],
             'max_depth':[50,100,200,None],
             'min_samples_split':[2,5,10,20]}

# use the best params found in above grid search to save time else it takes 2+ hrs for grid search
rfc_params_best = {'n_estimators':[400],
             'criterion':['entropy'],
             'max_depth':[400],
             'min_samples_split':[20]}

grid_rfc = GridSearchCV(RandomForestClassifier(class_weight='balanced',random_state=10), rfc_params_best, cv=5, 
                        scoring='accuracy')
grid_rfc.fit(X_train, y_train)




#print(grid_rfc.best_score_)
#print(grid_rfc.best_params_)




grid_rfc.score(X_test, y_test)




y_test_preds = grid_rfc.predict(X_test)
print(classification_report(y_test, y_test_preds))
confmat = confusion_matrix(y_test, y_test_preds)




fig,ax = plt.subplots(figsize=(3,3))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()




#TN,FP
#FN,TP

# FPR = 1 − specificity = FP / (FP + TN)
FPR = confmat[0,1]/(confmat[0,1]+confmat[0,0])

# FNR = 1 − sensitivity = FN / (TP + FN)
FNR = (confmat[1,0]/(confmat[1,1]+confmat[1,0]))

# error = (0.9*FNR) + (0.1*FPR)
error = (0.9*FNR) + (0.1*FPR)
print(error)




y_preds = pd.DataFrame(grid_rfc.predict(df_predict))




predictions = pd.concat((df_predict['new_ids'], y_preds), axis=1)




predictions.columns=['id', 'defective']




#predictions.head()




predictions.to_csv('./F02519.csv', index=False)

