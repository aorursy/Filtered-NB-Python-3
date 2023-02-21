#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




national_names = pd.read_csv("../input/NationalNames.csv")
national_names.info()




national_names["Year"].unique()




national_names.head()




mary = pd.DataFrame(national_names.loc[national_names["Name"] == "Mary"])
mary_f = pd.DataFrame(mary.loc[mary["Gender"] == "F"])
mary_f.head()

#sns.lmplot(data= national_names, x="Year", y="Count", row=national_names.loc[national_names["Name"] == "Mary")




mary_f.tail()




mary_f.plot(kind="scatter", x="Year", y="Count")




sammy = pd.DataFrame(national_names.loc[national_names["Name"] == "Sammy"])
sammy.head()




sns.FacetGrid(sammy, hue="Gender", size=5)    .map(plt.scatter, "Year", "Count")    .add_legend()




alex = pd.DataFrame(national_names.loc[national_names["Name"] == "Alex"])
alex.tail()




sns.FacetGrid(alex, hue="Gender", size=5)    .map(plt.scatter, "Year", "Count")    .add_legend()




bailey = pd.DataFrame(national_names.loc[national_names["Name"] == "Bailey"])
bailey.tail()




sns.FacetGrid(bailey, hue="Gender", size=5)    .map(plt.scatter, "Year", "Count")    .add_legend()




charlie = pd.DataFrame(national_names.loc[national_names["Name"] == "Charlie"])
charlie.tail()
sns.FacetGrid(charlie, hue="Gender", size=5)    .map(plt.scatter, "Year", "Count")    .add_legend()




taylor = pd.DataFrame(national_names.loc[national_names["Name"] == "Taylor"])
taylor.tail()
sns.FacetGrid(taylor, hue="Gender", size=5)    .map(plt.scatter, "Year", "Count")    .add_legend()

