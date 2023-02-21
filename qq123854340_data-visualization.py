#!/usr/bin/env python
# coding: utf-8



#This program is to visualize some features of original dataset

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

#First to see our dataset
voice = pd.read_csv("../input/voice.csv")
voice.head()




#check how many males and females separately in our dataset
voice["label"].value_counts()




#Scatter plot of given features
#You can compare other features by simply change "meanfun" and "meanfreq"
sns.FacetGrid(voice, hue="label", size=5)   .map(plt.scatter, "meanfun", "meanfreq")   .add_legend()
plt.show()




#Boxplot
#You can visualize other features by substituting "meanfun"
sns.boxplot(x="label",y="meanfun",data=voice)
plt.show()




#Distribution of male and female(every feature)
sns.FacetGrid(voice, hue="label", size=6)    .map(sns.kdeplot, "meanfun")    .add_legend()
plt.show()




#Radviz circle 
#Good to compare every feature
from pandas.tools.plotting import radviz
radviz(voice, "label")
plt.show()

