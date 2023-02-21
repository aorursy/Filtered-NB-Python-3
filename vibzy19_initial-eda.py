#!/usr/bin/env python
# coding: utf-8



import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from matplotlib.widgets import Cursor 
from matplotlib.ticker import FormatStrFormatter

import pandas as pd 
import scipy.stats as spstats
import gc

import tensorflow as tf

import os
print(os.listdir("../input"))

get_ipython().run_line_magic('matplotlib', 'inline')




crop_df = pd.read_csv("../input/apy.csv", encoding="utf-8")
cols = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"]




seasons = crop_df["Season"].unique()
seasons




states = crop_df["State_Name"].unique()
states




crops = crop_df["Crop"].unique()
years = np.sort(crop_df["Crop_Year"].unique())

print("Number of Crops : ", len(crops), ", Number of Seasons : ",len(seasons), ", Number of States : ", len(states), ", Number of Years : ", len(years))




crop_df.head()




state_df = crop_df

# Initialize
states_df = pd.DataFrame({"State":[state for state in states]}).set_index("State")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

for year in years:
    states_df[year] = [0 for state in states]

# Fill out
for state in states:
    for year in years:
        state_df = crop_df
        state_df = state_df[state_df.State_Name.str.contains(state) == True]
        state_df = state_df[state_df.Crop_Year.isin([year]) == True]
        s = state_df["Production"].sum()
        states_df.loc[state,year] = s 
states_df = states_df.T




# plot Bar Chart given the states

def plotBar(states):
    fig = plt.figure(figsize=(13,15))
    ax = fig.add_subplot(111)

    average = []
    stddevs = []

    for state in states:
        stddevs.append(np.std(np.asarray(states_df[state])))
        average.append(np.mean(states_df[state]))

    xAvg = [i for i in range(0,101) if i%2 == 0 or i == 0][:len(states)]

    cursor = Cursor(ax, useblit=True, color='red', linewidth=0.5)
    ax.set_title("Average Production acheived Standard Deviation", fontsize=15)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    plt.xlabel("State")
    plt.ylabel("Production")
    plt.xticks(xAvg, states,fontsize=12, rotation=90)
    plt.bar(xAvg, average, yerr=stddevs)


    plt.show()




# Plot Timeseries for each of the states based on the Production
def plotTimeseries(states,window):


    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111)

    def movingaverage (values, win):
        weights = np.repeat(1.0, win)/win
        sma = np.convolve(values, weights, 'valid')
        return sma

    NUM_COLORS = len(states)

    cm = plt.get_cmap('gist_rainbow')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

    ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    ax.set_title("Yearly Production of each of the states", fontsize=15)

    for state in states:
        ma = movingaverage(states_df[state],window)
        ax.plot(years[len(years)-len(ma):],ma,label=state)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.legend(loc='upper left')
    plt.show()




# Plot Timeseries for each of the states based on the Production

def plotBoxplot(states):    
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111)
    
    state_df = states_df[[state for state in states]]

    ax.boxplot(state_df.T)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))    
    ax.set_xticklabels(states, rotation=90)
    
    plt.legend(loc='upper left')
    plt.show()




plotTimeseries(states,3)




plotBar(states)




statesN = [state for state in states if state not in ["Kerala", "Andhra Pradesh", "Tamil Nadu"]]




plotTimeseries(statesN,3)




plotBar(statesN)




plotBoxplot(states)




plotBoxplot(statesN)

