#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')




x = pd.read_csv("../input/train.csv")




x.head()




len(x)




x.isnull().sum()




x.corr()




colormap = plt.cm.viridis
plt.figure(figsize=(25,25))
plt.title('Pearson Correlation of Features', y=1.05, size=20)
sns.heatmap(x.iloc[:,:-1].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)




train_data = x.iloc[:,:-1]
train_labels = x.iloc[:,-1]




train_data.head()




train_data_num = train_data._get_numeric_data()









train_labels.head()




test_data = pd.read_csv("../input/test.csv")




test_data.head()




len(test_data)




test_data = y




ssub = pd.read_csv("../input/sample_submission.csv")




ssub.head()




from sklearn.model_selection import cross_val_predict
from sklearn import linear_model




without_missing = train_data_num.isnull() == 0
train_data_num = train_data_num.dropna(axis="columns")




train_data_num.isnull().sum()




train_data_num.head()




lr = linear_model.LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, train_data_num, train_labels, cv=10)





fig, ax = plt.subplots()
ax.scatter(train_labels, predicted)
ax.plot([train_labels.min(), train_labels.max()], [train_labels.min(), train_labels.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()




colnames_train = list(train_data_num)




test_data = test_data[colnames_train]









test_data.isnull().sum()




# test_data contains NaN values. Impute the mean.
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=1)

test_data_no_nan = pd.DataFrame(imp.fit_transform(test_data))
test_data_no_nan.columns = test_data.columns
test_data_no_nan.index = test_data.index




test_data_no_nan.isnull().sum()




test_data_no_nan.tail()




# Train the model using the training sets
lr.fit(train_data_num, train_labels)
test_labels = lr.predict(test_data_no_nan)




# The coefficients
print('Coefficients: \n', lr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((lr.predict(test_data_no_nan) - test_labels) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr.score(test_data_no_nan, test_labels))




len(ssub)




len(test_labels)




test_labels[1:10]




ssub.tail()




new_sub = ssub
new_sub["SalePrice"] = test_labels




new_sub.head()




new_sub.isnull().sum()




invalid = new_sub["SalePrice"] < 0
new_sub[invalid]




new_sub[invalid] = 0




invalid = new_sub["SalePrice"] < 0
new_sub[invalid]




len(new_sub)




new_sub.to_csv("./submission.csv", index=False)




print(check_output(["ls", "./"]).decode("utf8"))









x2 = pd.read_csv("../input/train.csv")




x2.head()




# x2[::, x2.isnull().sum() > 0]
x2.isnull().sum()




no_missing = x2.columns[x2.isnull().sum() == 0]




x2_no_miss = x2.loc[:,no_missing]
train_data = x2_no_miss.iloc[:, :-1]
train_labels = x2_no_miss.iloc[:, -1]
train_labels[0:5]




train_data.head()




numeric_colnames = train_data._get_numeric_data().columns
numeric_colnames




len(numeric_colnames)




non_numeric_colnames = [colname for colname in train_data.columns if colname not in numeric_colnames]
non_numeric_colnames




len(non_numeric_colnames)




print("Total columns without missing values: {}".format(len(train_data.columns)))
print("Columns with numeric values: {}".format(len(numeric_colnames)))
print("Columns with non-numeric values: {}".format(len(non_numeric_colnames)))




numeric_train_data = train_data.loc[:, numeric_colnames]
non_numeric_train_data = train_data.loc[:, non_numeric_colnames]




# just verifying shapes
print("numeric_train_data: {}".format(numeric_train_data.shape))
print("non_numeric_train_data: {}".format(non_numeric_train_data.shape))




# write function
# def dummify_what_you_can(df):
#     dummify all non_numeric columns
#     drop the original columns
#     so you're left with the numeric + dummy columns
def dummify_all_non_numeric(df):
    import pandas as pd
    num_df = df._get_numeric_data()
    numeric_colnames = num_df.columns
    non_numeric_colnames = [colname for colname in df.columns if colname not in numeric_colnames]
    non_num_df = df.loc[:, non_numeric_colnames]
    dummy_df = pd.get_dummies(non_num_df, prefix=non_numeric_colnames)
    return pd.concat([num_df, dummy_df], axis=1)




train_data_dum = dummify_all_non_numeric(train_data)
train_data_dum.head()




train_data_dum.shape




train_data_colnames = train_data_dum.columns




test_data = pd.read_csv("../input/test.csv")




test_data.head()




test_data_dum = dummify_all_non_numeric(test_data)
test_data_dum = test_data_dum.dropna(axis='columns')




from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

lr = linear_model.LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, train_data_dum, train_labels, cv=10)

# Train the model using the training sets
lr.fit(train_data_dum, train_labels)
test_labels = lr.predict(test_data_dum)

# The coefficients
print('Coefficients: \n', lr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((lr.predict(test_data_dum) - test_labels) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lr.score(test_data_no_nan, test_labels))






