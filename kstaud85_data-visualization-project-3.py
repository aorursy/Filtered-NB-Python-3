#!/usr/bin/env python
# coding: utf-8




# To begin, import the following: pandas, seaborn, and bokeh
# To ignore warnings, use the following code to make the display more attractive.
import pandas as pd
import bokeh
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
sns.set(style="white", color_codes=True)

#To import the pollution dataset:
pollution= pd.read_csv("../input/pollution_us_2000_2016.csv")


#To view income data:
pollution.head()






#view the number of observations (rows and columns)
pollution.shape




#check the type of data we have
type(pollution)




#To check for any missing values
pollution.isnull().any()




#Some values are missing
#drop the columns that have missing data 
pollution.dropna(inplace=True)
pollution.reset_index(inplace=True, drop=True)
pollution.isnull().any()




#To view the number of monitoring sites by state:
pollution["State"].value_counts()




#drop unwanted columns
data=pollution.drop(['State Code', 'County Code','Site Num','Address','County','City','NO2 Units','NO2 1st Max Value','NO2 1st Max Hour','O3 Units','O3 1st Max Value','O3 1st Max Value','SO2 Units','SO2 1st Max Value','SO2 1st Max Hour','CO Units','CO 1st Max Value','CO 1st Max Hour','Unnamed: 0'], axis=1)




#Now we have the data we want to start graphing
data.head()




#Look at the shape (rows and columns) of data, again
data.shape




#Now we are ready to create some data visualizations
#This seaborn barplot shows the mean NO2 levels for each state
ax = sns.barplot(x="NO2 Mean",y="State", data=data, ci=100)
fig = plt.gcf()
fig.set_size_inches(30, 13)




#Repeat this step for the O3 Mean
ax = sns.barplot(x="O3 Mean",y="State", data=data, ci=100)
fig = plt.gcf()
fig.set_size_inches(18, 13)
plt.show()




#Repeat this step for the O3 Mean
ax = sns.barplot(x="O3 Mean",y="State", data=data, ci=100)
fig = plt.gcf()
fig.set_size_inches(18, 13)




#Now we will plot the SO2 Mean with the same graph to show differences in pollutant levels
ax = sns.barplot(x="SO2 Mean",y="State", data=data, ci=100)
fig = plt.gcf()
fig.set_size_inches(18, 13)




#This shows the total mean data grouped by state
data.groupby('State').mean().plot(kind = 'bar', figsize = (20, 13))




#To create a seaborn plot using NO2 Mean and SO2 Mean Data
sns.FacetGrid(data, hue="State", size=5)    .map(plt.scatter, "SO2 Mean", "NO2 Mean")    .add_legend()
plt.show()




#create a scatterplot in bokeh
#image can be seen as an at the link below
from bokeh.charts import Scatter, output_file, show

p = Scatter(data, x='State', y='SO2 AQI', color='blue', title="SO2 AQI by State",
            xlabel="State", ylabel="SO2 AQI",height=1000, width=1000)

output_file("pollution.html")

show(p)






