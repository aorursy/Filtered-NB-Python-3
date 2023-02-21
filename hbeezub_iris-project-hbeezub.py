#!/usr/bin/env python
# coding: utf-8



import pandas as pd




import warnings 
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="white", color_codes=True)




iris = pd.read_csv("../input/Iris.csv")




iris.head()




iris["Species"].value_counts()




iris["PetalWidthCm"].value_counts()




iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")




sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=7)




sns.FacetGrid(iris, hue="Species", size=5,palette="RdBu_d")    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")   .add_legend()




sns.boxplot(x="Species", y="PetalLengthCm", data=iris,palette="Blues")




ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris,palette="Blues")
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris,palette="Blues_d", jitter=True, edgecolor="gray")




ax = sns.violinplot(x="Species", y="PetalLengthCm", data=iris,palette="Blues", size=6)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris,palette="Blues_d", jitter=True)




sns.FacetGrid(iris, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()




sns.pairplot(iris.drop("Id", axis=1), hue="Species",palette="Greens", size=3)




sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")




iris.drop("Id", axis=1).boxplot(by="Species", figsize=(10, 10))




from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")




from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")




from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")

