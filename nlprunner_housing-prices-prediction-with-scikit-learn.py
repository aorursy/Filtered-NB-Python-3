#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as metrics




df = pd.read_csv('../input/train.csv')
df = df.set_index('Id')

sdf = pd.read_csv('../input/test.csv')
sdf = sdf.set_index('Id')
df.head()
print (df.columns.values)




y = df.SalePrice
print("Average sale price: " + "${:,.0f}".format(y.mean()))




df = df.drop('SalePrice', axis=1)
all_df = df.append(sdf)
all_df.shape




all_features = list(df.columns.values)
numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','TotalBsmtSF','Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
#numeric_features = list(df.select_dtypes(include=[np.number]).columns.values)

(len(all_features), len(categorical_features), len(numeric_features))




numeric_df = all_df[numeric_features]
numeric_df.shape




X = numeric_df.as_matrix()

imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X)
X = imp.transform(X)
X.shape




scaler = pp.StandardScaler()
#Todo: Fit and transform data using scaler


X[0, :]




def process_categorical(ndf, df, categorical_features):
    for f in categorical_features:
        new_cols = pd.DataFrame(pd.get_dummies(df[f]))
        new_cols.index = df.index
        ndf = pd.merge(ndf, new_cols, how = 'inner', left_index=True, right_index=True)
    return ndf

numeric_df = pd.DataFrame(X)
numeric_df.index = all_df.index
combined_df = process_categorical(numeric_df, all_df, categorical_features)
print(df['SaleCondition'].head())
print (set(df["SaleCondition"].values))
print(combined_df[['AdjLand', 'Family', 'Partial', 'Abnorml', 'Normal', 'Alloca']].head())




X = combined_df.as_matrix()
X.shape




#PCA
from sklearn.decomposition import PCA

test_n = df.shape[0]
x = X[:test_n,:]

pca = PCA()
#Todo: Fit and transform X using PCA (function params: training data and labels)


X.shape




x = X[:test_n,:]
x_test = X[test_n:,:]
#Todo: split training data up into training and validation sets




from sklearn import linear_model

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)





print('Linear Regression score is %f' % lr.score(x_val, y_val))
print('Ridge score is %f' % ridge.score(x_val, y_val))




y_submit = classifier.predict(x_test)
y_submit[y_submit < 0] = 1.
sdf['SalePrice'] = y_submit
sdf.to_csv('./submission.csv', columns = ['SalePrice'])

