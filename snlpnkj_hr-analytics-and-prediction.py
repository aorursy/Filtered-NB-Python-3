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




#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




data = pd.read_csv('../input/HR_comma_sep.csv')
data.head() # will give top five rows 




data.tail() # give last five rows 




data.shape # print shape(row, column)




data.describe()




data.info()  # check what kind of datas are 




# check how many salary types we have 
data.groupby('salary').size()




data.isnull().values.any()  # check data has any values null/nan or not




column_names = data.columns.tolist()
print column_names




data[data['left'] == 1].shape[0]  # employees are left 




data[data['left'] == 0].shape[0]# are still working




# Why employees are leaving?
a = data[data['left']==1].mean()
a




b = data[data['left'] == 0].mean()
b




# make prediction of left 




# see here:  https://github.com/sunilpankaj/Human-Resources-Analytics-project-using-python-






