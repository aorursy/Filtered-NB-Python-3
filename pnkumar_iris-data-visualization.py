#!/usr/bin/env python
# coding: utf-8



# This visualization was created for personal education and have reused many codes from other kernels 

# Let's import python first as it an important package for data processing
import pandas as pd

# This method produces a lot of warning, so let's import the package and ignore those warnings 
import warnings 
warnings.filterwarnings("ignore")

# we need seaborn as it's a Python graphing library
import seaborn as sns

# We need matplot to plot the data in for Iris dataset
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# let's load the iris data set using pandas
iris = pd.read_csv("../input/Iris.csv") 

# This handles  the error in jyputer for not plotting
get_ipython().run_line_magic('matplotlib', 'inline')

# Lets us see the first 10 entries of the file.
iris.head(10)




# Count the column  Species and see how many we have in each category
iris["Species"].value_counts()




# Plotting a scattered plot with SepalLengthCm on the x-axis and SepalWidthCm on the y-axis for this we will use the .plot command
# We see that we don't get much information with this graph
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm",color='#800000',linewidth=3)




# We see that scatter plot doesn't give enough information, so we try line plot
iris.plot(kind="line", x="SepalLengthCm", y="SepalWidthCm",color='#800000',linewidth=2,          linestyle='dashed',style='-->--',)




# We'll use seaborn's FacetGrid to color the scatterplot by species and also add the legend
# We will use \ command to move the code to next line so it is readable
sns.FacetGrid(iris, hue="Species", size=6)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()




# we will use seaborn jointplot shows bivariate scatterplots and univariate histograms with regression 
# line density estimation in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=6, kind='reg', color='#800000')




# we will use seaborn jointplot shows bivariate scatterplots and univariate histograms with Kernel density 
# estimation in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=6, kind='kde', color='#800000', space=0)




# We will use seaborn jointplot shows bivariate scatterplots and univariate histograms with hex graph in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=6, kind='hex', color='#800000')




# We can use boxplot which is standardized way of displaying the distribution of data based 
# on the five-number summary: minimum, first quartile, median, third quartile, and maximum.
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)




# We try to strip the plot and use jitter to see all the point for each species and save it in axes as ax
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")




# We use swarmplot to plot the data
sns.swarmplot(x="SepalLengthCm", y="SepalWidthCm", data=iris)




# We use violinplot to plot the data
sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)




# We see that only swarmplot and violinplot is not able to get enough information in last two plot.
# So we use violet plot, The violin plot is similar to box plots, except that they also show 
# the probability density of the data at different values
sns.swarmplot(x="Species", y="SepalLengthCm", data=iris,palette='Reds')
sns.violinplot(x="Species", y="SepalLengthCm", data=iris, palette='muted', inner=None)




# Seaborn plot is useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(iris, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()




# We will use pairplot which shows relation between each pair
# we observe how the species Iris Setosa, which is blue is separated from others
# We see that Petal length and petal width are correlated
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)




# We will use pairplot which shows relation between each pair and also kernel density estimate
sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")




# We plot boxplot using panda
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))




# We use Andrew curves using panda for the three species
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")




# We use Parallel coordinates using panda for the three species
# Parallel coordinates plot each feature on a separate column & then draws lines
# connecting the features for each data sample
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")




# we use multivariate visualization technique pandas has is radviz
# It enables the visualization of multidimensional data while maintaining the relation to the original dimensions.
from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")

