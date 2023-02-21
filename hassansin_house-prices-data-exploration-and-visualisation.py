#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

from sklearn.linear_model import LinearRegression




houses=pd.read_csv("../input/train.csv")
houses.head()




houses_test = pd.read_csv("../input/test.csv")
#transpose
houses_test.head()
#note their is no "SalePrice" column here which is our target varible.




#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset
#(rows,columns)
houses.shape




houses_test.shape
#1 column less because target variable isn't there in test set!




#info method provides information about dataset like 
#total values in each column, null/not null, datatype, memory occupied etc
houses.info()




#How many columns with different datatypes are there?
houses.get_dtype_counts()




##Describe gives statistical information about numerical columns in the dataset
houses.describe()




corr=houses.corr()["SalePrice"]
corr[np.argsort(corr, axis=0)[::-1]]




corr=houses[['OverallQual' ,'GrLivArea' ,'GarageCars','GarageArea' ,
            'TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd',
            'YearBuilt','YearRemodAdd','SalePrice']].corr()#["SalePrice"]
#plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=1, square=True,annot=True)
plt.title('Correlation between features')




correlations=houses.corr()
attrs = correlations.iloc[:-1,:-1] # all except target

threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0])     .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), 
        columns=['Attribute Pair', 'Correlation'])

    # sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

unique_important_corrs




houses[['OverallQual','SalePrice']].groupby(['OverallQual'],
as_index=False).mean().sort_values(by='OverallQual', ascending=False)




houses[['GarageCars','SalePrice']].groupby(['GarageCars'],
as_index=False).mean().sort_values(by='GarageCars', ascending=False)




houses[['Fireplaces','SalePrice']].groupby(['Fireplaces'],
as_index=False).mean().sort_values(by='Fireplaces', ascending=False)




#lets see if there are any columns with missing values 
null_columns=houses.columns[houses.isnull().any()]
houses[null_columns].isnull().sum()




houses['LotFrontage'].corr(houses['LotArea'])




houses['SqrtLotArea']=np.sqrt(houses['LotArea'])
houses['LotFrontage'].corr(houses['SqrtLotArea'])




sns.jointplot(houses['LotFrontage'],houses['SqrtLotArea'],color='gold')




filter = houses['LotFrontage'].isnull()
houses.LotFrontage[filter]=houses.SqrtLotArea[filter]




plt.scatter(houses["MasVnrArea"],houses["SalePrice"])
plt.show()




houses["MasVnrType"].value_counts().plot(kind='bar',colors='gold')




houses["MasVnrType"] = houses["MasVnrType"].fillna('None')
houses["MasVnrArea"] = houses["MasVnrArea"].fillna(0.0)




labels = houses["Electrical"].unique()
sizes = houses["Electrical"].value_counts().values
explode=[0.1,0,0,0,0]
parcent = 100.*sizes/sizes.sum()
labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]

colors = [  'lightcoral','gold','yellowgreen', 'lightblue','purple']
patches, texts= plt.pie(sizes, colors=colors,explode=explode,
                        startangle=90)
plt.legend(patches, labels, loc="best")

plt.title("Electrical")
plt.show()




#We can replace missing values with most frequent ones.
houses["Electrical"] = houses["Electrical"].fillna('SBrkr')




sns.stripplot(x=houses["Alley"], y=houses["SalePrice"],jitter=True)




houses["Alley"] = houses["Alley"].fillna('None')




sns.distplot(houses["TotalBsmtSF"])




basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
houses[basement_cols][houses['BsmtQual'].isnull()==True]




for col in basement_cols:
    if 'FinSF'not in col:
        houses[col] = houses[col].fillna('None')




sns.factorplot("Fireplaces","SalePrice",data=houses,hue="FireplaceQu")




#If fireplace quality is missing that means that house doesn't have a fireplace
houses["FireplaceQu"] = houses["FireplaceQu"].fillna('None')
pd.crosstab(houses.Fireplaces, houses.FireplaceQu)




sns.boxplot(houses["GarageArea"],color='r')




sns.barplot(houses["GarageCars"],houses["SalePrice"])




garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
houses[garage_cols][houses['GarageType'].isnull()==True]




for col in garage_cols:
    if houses[col].dtype==np.object:
        houses[col] = houses[col].fillna('None')
    else:
        houses[col] = houses[col].fillna(0)




#If PoolArea is 0, that means that house doesn't have a pool.
#So we can replace PoolQuality with None.
houses["PoolQC"] = houses["PoolQC"].fillna('None')




labels = houses["Fence"].unique()
sizes = houses["Fence"].value_counts().values
explode=[0.1,0,0,0]
parcent = 100.*sizes/sizes.sum()
#labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]

colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral']
patches, texts,autotexts= plt.pie(sizes, colors=colors,autopct="%1.1f%%",
                        shadow=True,startangle=90)
plt.legend(patches, labels, loc="best")

plt.title("Fence")
plt.show()




houses["Fence"] = houses["Fence"].fillna('None')




houses["MiscFeature"].value_counts().plot(kind='pie')




#Some houses don't have miscellaneous features like shed, Tennis court etc..
houses["MiscFeature"] = houses["MiscFeature"].fillna('None')




#Let's confirm that we have removed all missing values
houses[null_columns].isnull().sum()




labels = houses["MSZoning"].unique()
sizes = houses["MSZoning"].value_counts().values
explode=[0.1,0,0,0,0]
parcent = 100.*sizes/sizes.sum()
labels = ['{0} - {1:1.1f} %'.format(i,j) for i,j in zip(labels, parcent)]

colors = ['yellowgreen', 'gold', 'lightblue', 'lightcoral','blue']
patches, texts= plt.pie(sizes, colors=colors,explode=explode,
                        shadow=True,startangle=90)
plt.legend(patches, labels, loc="best")

plt.title("Zoning Classification")
plt.show()




plt.hist(houses["GrLivArea"],color='purple')
plt.xlabel("Ground Living Area")
plt.show()




sns.distplot(houses["YearBuilt"],color='seagreen')




sns.distplot(houses["YearRemodAdd"],color='r')




houses["Heating"].value_counts().plot(kind='bar')




sns.factorplot('HeatingQC', 'SalePrice', hue = 'CentralAir', data=houses)




sns.countplot(houses["FullBath"])




sns.countplot(houses["TotRmsAbvGrd"])




sns.factorplot("KitchenAbvGr","SalePrice",data=houses,hue="KitchenQual")




sns.countplot(x = 'Neighborhood', data = houses)
plt.xticks(rotation=45) 
plt.show()




plt.barh(houses["OverallQual"],width=houses["SalePrice"],color="r")




plt.scatter(houses["1stFlrSF"],houses["SalePrice"])
plt.show()




#street
sns.stripplot(x=houses["Street"], y=houses["SalePrice"])






