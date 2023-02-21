#!/usr/bin/env python
# coding: utf-8



# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)




# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")




# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")




original_data




print max(original_data)
print min(original_data)




sns.distplot(original_data, bins=9, kde=False, rug=True)




sns.rugplot(original_data)




# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")




print(min(usd_goal))
print(max(usd_goal))

print(usd_goal.shape)




# a close look to the graph of original data

fig, ax = plt.subplots(2,1)
# ax[0].set_xlim([0, 5000000])
ax[0].set_ylim([0, 100])
sns.distplot(usd_goal, kde=False, ax=ax[0])
ax[0].set_title("limited y-axis")

sns.distplot(usd_goal, kde=False, ax=ax[1])
# ax[1].set_title("original graph")




# a close look to the graph of scaled data

fig, ax = plt.subplots(2,1)
# ax[0].set_xlim([0, 5000000])
ax[0].set_ylim([0, 100])
sns.distplot(scaled_data, kde=False, ax=ax[0])
ax[0].set_title("limited y-axis")

sns.distplot(scaled_data, kde=False, ax=ax[1])
# ax[1].set_title("original graph")




# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?




goal = kickstarters_2017.goal

scaled_data = minmax_scaling(goal, columns=[0])

fig, ax = plt.subplots(1,2)
sns.distplot(goal, ax=ax[0])
ax[0].set_title("original data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("scaled data")




print(min(usd_goal))
print(max(usd_goal))

print(usd_goal.shape)




# a close look to the graph of original data

fig, ax = plt.subplots(2,1)
# ax[0].set_xlim([0, 5000000])
ax[0].set_ylim([0, 100])
sns.distplot(goal, kde=False, ax=ax[0])
ax[0].set_title("limited y-axis")

sns.distplot(goal, kde=False, ax=ax[1])
# ax[1].set_title("original graph")




t = kickstarters_2017.usd_pledged_real
print(t.shape)
k = kickstarters_2017.usd_pledged_real > 0
print(k.shape)




# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")




# Your turn! 
# We looked as the usd_pledged_real column. What about the "pledged" column? Does it have the same info?




t = kickstarters_2017.pledged
print(t.shape)
k = kickstarters_2017.pledged > 0
print(k.shape)




index_of_positive_pledges = kickstarters_2017.pledged > 0

positive_pledges = kickstarters_2017.pledged.loc[index_of_positive_pledges]

normalized_pledges = stats.boxcox(positive_pledges)[0]

fig, ax = plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("original data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("normalized data")

