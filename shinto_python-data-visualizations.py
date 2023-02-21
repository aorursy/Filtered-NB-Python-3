#!/usr/bin/env python
# coding: utf-8



# First , we'll import pandas. a data processing and CSV file I/O library
import pandas as pd

# We'l also import seaborn, a Python graphing library
import warnings # current version of searborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the Iris flower dataset, which is in the "../input/" directory
iris = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame

# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do 
iris.head()
# Press shift+enter to execute this cell




# Let's see how many examples we have of each species
iris['Species'].value_counts()




# The first way we can plot this is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the Iris features.
iris.plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm')




# We can also use the seaborn library to make a similar plot
# A seaborn joint plot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=iris, size=5)




# One piece of information missing in the plots above is what species each plant is 
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(iris, hue='Species', size=5)              .map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm')              .add_legend()




# We can look at an individual feature through a boxplot
sns.boxplot(x='Species', y='PetalLengthCm', data=iris)




# One way we can extend this plot is addina a layer of individual points on top of
# it through Seaborn's striplot

# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species

# Saving the resulting axes as ax each time causes the resulting  plot to be shown
# on top of the previous axes
ax = sns.boxplot(x='Species', y='PetalLengthCm', data=iris)
ax = sns.stripplot(x='Species', y='PetalLengthCm', data=iris, jitter=True, edgecolor='gray')




iris.describe()




# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data fatter, and sparser thinner in a violin plot
sns.violinplot(x='Species', y='PetalLengthCm', data=iris, size=6)




# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# wchich creates and visualizes a kernel density estimate of the underlying feature
sns.FacetGrid(iris, hue='Species', size=6)     .map(sns.kdeplot, 'PetalLengthCm')     .add_legend()




# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features

# From the pairplot, we'll see that the Iris-setosa species is separated from the other 
# two across all features combinations
sns.pairplot(iris.drop('Id',axis=1), hue='Species', size=3)




# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde
sns.pairplot(iris.drop('Id', axis=1), hue='Species', size=3, diag_kind='kde')




# Now that we have covered seaborn, Let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species
iris.drop('Id', axis=1).boxplot(by='Species', figsize=(12, 6))




# One cool more sophisticated technique pandas has available is called Andrews Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
from pandas.tools.plotting import andrews_curves
andrews_curves(iris.drop('Id', axis=1), 'Species')




# Another multivariate visualization techique pandas has is parallel coordinates
# Parallel coordiates plots each feature on a separate column & then draws lines
# connecting the features for each data sample
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop('Id', axis=1), 'Species')




# A final multivariate visualization technique pandas has is radviz
# which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(iris.drop('Id', axis=1), 'Species')

