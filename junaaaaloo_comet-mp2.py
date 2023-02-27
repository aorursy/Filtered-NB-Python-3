#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import collections as co
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




# Read whole data set to a variable 
vgsales = pd.read_csv("../input/Video_Game_Sales_as_of_Jan_2017.csv")
vgsales = vgsales.dropna()




x = vgsales['User_Score']
y = vgsales['Global_Sales']

plt.scatter(x, y)
plt.show()




samp = {"User Score": vgsales['User_Score'], "Global Sales": vgsales['Global_Sales']}
dsamp = pd.DataFrame(samp)
dsamp.corr('pearson')




years = vgsales.groupby(['Year_of_Release']).count()
years




# <h4> <b> Question 3: </b> Which game genre is the most popular? </h4>




vgsales.groupby(['Genre']).count()['Name']

