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




genderclassmodel = pd.read_csv("../input/genderclassmodel.csv")
gendermodel = pd.read_csv("../input/gendermodel.csv")
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

print(pd.concat([genderclassmodel.head(), genderclassmodel.tail()]))
#print('-'*40)
#print(pd.concat([gendermodel.head(), gendermodel.tail()])) # same as genderclassmodels
print()
print('-'*20, "train data", '-'*20)
print(pd.concat([train.head(), train.tail()]))




# train.columns.values
for col in train.columns:
    print(col, '\t', train[col].dtype)
    #col_name = col.str()
    
   
print (train.Cabin.unique())
    
#train[train['Cabin'].str.contains("NaN")==True] #can only .str an object typed data
##df[df['A'].str.contains("Hello|Britain")==True]

