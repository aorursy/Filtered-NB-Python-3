#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Some appearance options.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")
display(data.columns)




linear_model_formula = " 'number_people' ~ 'timestamp'+ 'day_of_week'+ 'is_weekend'+ 'is_holiday'+'apparent_temperature'"
linear_model = smf.ols(formula=linear_model_formula, data=data)
linear_model_fit = linear_model.fit()
linear_model_fit.summary()

