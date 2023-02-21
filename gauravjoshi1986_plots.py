#!/usr/bin/env python
# coding: utf-8



import pandas as pd

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the Iris flower dataset, which is in the "../input/" directory
iris = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame

# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do
iris.head()




iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")




sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)




# With Color codes
sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()




ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")




sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)




iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))






