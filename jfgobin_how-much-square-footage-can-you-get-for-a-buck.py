#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# First, some cleaning and conversions
# Conversion of categorical to type "category" and replacement of
# missing values in categorical with "missing"
for i_name in ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
               'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
               'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
               'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
               'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
               'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
               'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir',
               'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
               'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
               'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']:
        train_data[i_name].fillna("missing", inplace=True)
        train_data[i_name] = train_data[i_name].astype('category')
        test_data[i_name] = test_data[i_name].astype('category')
# Let's look at each categorical




train_data['SFPerBuck'] = (train_data['GrLivArea'])/train_data['SalePrice']
sns.distplot(train_data['SFPerBuck'])
train_data['SFPerBuck'].describe()




#skewness and kurtosis
print("            {:^9}  {:^9}".format("SalePrice", "SFPerBuck"))
print("Skewness:   {:9.6F}  {:9.6F}".format(train_data['SalePrice'].skew(),
                                            train_data['SFPerBuck'].skew()))
print("Kurtosis:   {:9.6F}  {:9.6F}".format(train_data['SalePrice'].kurt(),
                                            train_data['SFPerBuck'].kurt()))




log_stat = np.log(train_data['SFPerBuck'])
print("            {:^9}  {:^9}".format("SalePrice", "SFPerBuck"))
print("Skewness:   {:9.6F}  {:9.6F}".format(np.log(train_data['SalePrice']).skew(),
                                            np.log(train_data['SFPerBuck']).skew()))
print("Kurtosis:   {:9.6F}  {:9.6F}".format(np.log(train_data['SalePrice']).kurt(),
                                            np.log(train_data['SFPerBuck']).kurt()))




sns.distplot(log_stat, fit=norm);
fig = plt.figure()
res = stats.probplot(log_stat, plot=plt)
train_data['logSFPerBuck'] = log_stat




#correlation matrix
corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);




#saleprice correlation matrix
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'logSFPerBuck')['logSFPerBuck'].index
cm = np.corrcoef(train_data[cols].values.T)
f, ax = plt.subplots(figsize=(8, 8))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 
                 fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, 
                 xticklabels=cols.values)
plt.show()

cols = corrmat.nsmallest(k, 'logSFPerBuck')['logSFPerBuck'].index
cols = cols.insert(0, "logSFPerBuck")
cm = np.corrcoef(train_data[cols].values.T)
f, ax = plt.subplots(figsize=(8, 8))
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, 
                 fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, 
                 xticklabels=cols.values)
plt.show()




train_data['OverallQual'].describe()




train_data['OverallQual'].value_counts()




fig = sns.distplot(train_data['OverallQual'])




train_data["QualBinned"] = pd.cut(train_data['OverallQual'],[0,5.5,6.5,10], labels=["Low","Medium","High"])
fig = sns.boxplot(x="QualBinned", y="logSFPerBuck", data=train_data)




train_data['OverallCond'].describe()




train_data['OverallCond'].value_counts()




fig = sns.distplot(train_data['OverallCond'])




train_data["CondBinned"] = pd.cut(train_data['OverallCond'],[0,4.5,5.5,10], labels=["Low","Medium","High"])
fig = sns.boxplot(x="CondBinned", y="logSFPerBuck", data=train_data)




f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="Neighborhood", y="logSFPerBuck", data=train_data)
fig.axis(ymin=-6, ymax=-3);
plt.xticks(rotation=90);




train_data.loc[train_data['Neighborhood']=="Blueste"]




f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="Neighborhood", y="logSFPerBuck", data=train_data, hue="QualBinned")
fig.grid()
fig.axis(ymin=-6, ymax=-3);
plt.xticks(rotation=90);




data_gilbert = train_data.loc[train_data['Neighborhood']=="Gilbert"]
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="OverallQual", y="logSFPerBuck", data=data_gilbert)
fig.grid()
fig.axis(ymin=-5.5, ymax=-4.0);




f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="MSZoning", y="logSFPerBuck", data=train_data)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);




f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="MSSubClass", y="logSFPerBuck", data=train_data)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);

f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="HouseStyle", y="logSFPerBuck", data=train_data)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);




ms_hs = pd.DataFrame(train_data.groupby(['MSSubClass', 'HouseStyle'])['HouseStyle'].count())
ms_hs.columns=['Count']
ms_hs




f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="BldgType", y="logSFPerBuck", data=train_data)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);




f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="Condition1", y="logSFPerBuck", data=train_data, hue="Condition2")
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);




train_data['C1C2'] = train_data['Condition1'].astype("str") +                      ','+train_data['Condition2'].astype("str")
    
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="C1C2", y="logSFPerBuck", data=train_data)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);
plt.xticks(rotation=90);




f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="SaleCondition", y="logSFPerBuck", data=train_data, hue="SaleType")
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);
plt.xticks(rotation=90);




fig = sns.jointplot(x="YearBuilt", y="logSFPerBuck", data=train_data, 
                    size=12, kind="reg")




train_data.query("YearBuilt<1946 and MSSubClass==20")[["Id","MSSubClass","YearBuilt","YearRemodAdd"]]




train_data.query("YearBuilt>1945 and MSSubClass==30")[["Id","MSSubClass","YearBuilt","YearRemodAdd"]]




oneStory = train_data.query("MSSubClass in (20,30,40,120)")
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="MSSubClass", y="logSFPerBuck", data=oneStory)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);
fig = sns.jointplot(x="YearBuilt", y="logSFPerBuck", data=oneStory, 
                    size=12, kind="reg")




train_data.query("YearBuilt<1946 and MSSubClass==160")[["Id","MSSubClass","YearBuilt","YearRemodAdd"]]




fig = sns.jointplot(x="YearBuilt", y="YearRemodAdd", data=train_data, 
                    size=12, kind="reg")




play_data = train_data[['YearBuilt','YearRemodAdd']].copy()
play_data.loc[play_data['YearRemodAdd']==1950,'YearRemodAdd'] = 0
def seriesmax(argseries):
    return argseries.max()
YearMaxRAB = play_data[['YearBuilt','YearRemodAdd']].apply(seriesmax, axis=1)
train_data['YearMaxRAB'] = YearMaxRAB
fig = sns.jointplot(x="YearBuilt", y="YearMaxRAB", data=train_data, 
                    size=12, kind="reg")




corrmat = train_data[['logSFPerBuck','YearBuilt','YearRemodAdd',
                      'YearMaxRAB', 'SalePrice', 'SFPerBuck']].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, annot=True, square=True, 
                 fmt='.2f', annot_kws={'size': 8});




fig = sns.jointplot(x="YrSold", y="logSFPerBuck", data=train_data, 
                    size=12, kind="reg")




surfvar = ['logSFPerBuck', 'SFPerBuck', 'SalePrice', 'LotArea', 'MasVnrArea',
           'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GrLivArea',
           'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
           'ScreenPorch', 'PoolArea', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF']

corrmat = train_data[surfvar].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, annot=True, square=True, 
                 fmt='.2f', annot_kws={'size': 8});




fig = sns.jointplot(x="3SsnPorch", y="logSFPerBuck", data=train_data, 
                    size=12, kind="reg")




TSsnPorchPresent = train_data['3SsnPorch'] > 0
train_data['3SsnPorchPresent'] = TSsnPorchPresent.astype('category')
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="3SsnPorchPresent", y="logSFPerBuck", data=train_data)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);




whichvar = 'LotFrontage'
fig = sns.jointplot(x=whichvar, y="logSFPerBuck", data=train_data, 
                    size=12, kind="reg")

varPresent = train_data[whichvar] > 0
train_data['varPresent'] = varPresent.astype('category')
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="varPresent", y="logSFPerBuck", data=train_data)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);




f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x="MiscFeature", y="logSFPerBuck", data=train_data)
fig.grid()
fig.axis(ymin=-6.0, ymax=-3.0);

