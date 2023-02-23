#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




train_df = pd.read_csv('../input/train.csv') 
test_df = pd.read_csv('../input/test.csv')




print(train_df.info())
print("----------------------------")
test_df.info()




train_df.head(5)
test_df.head(5)




# We should drop PassengerId for our predictive model




train_df.drop("PassengerId",axis=1,inplace=True)
train_df.head(5)




train_df["Survived"].value_counts(normalize=True)




# Around 62% of the people in the training set died






