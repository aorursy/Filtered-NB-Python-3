#!/usr/bin/env python
# coding: utf-8



# Import the required modules
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
iris = pd.read_csv("../input/Iris.csv")

# Lets store the Species subset into outcomes variable for future use
outcomes = iris['Species']

# Lets have a glimpse of first few datapoints of the iris dataset
iris.head()




""" A model with no features. Always predicts the species as Isis-setosa"""
def predictions_0(data):
    predictions = []
    for _, species in data.iterrows():
        # predict as Iris-setosa for every species
        predictions.append('Iris-setosa')
        
    # return the predictions
    return pd.Series(predictions)

predictions = predictions_0(iris)




print(accuracy_score(outcomes, predictions))




# Plot a scatterplot
sb.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()




# A model with few features.
def predictions_1(data):
    predictions = []
    for _, species in data.iterrows():
        
        # Convert strings to float and store them with shorter names
        sw = float(species['SepalWidthCm'])
        sl = float(species['SepalLengthCm'])
        if (sw > 2.8) & (sl < 5.5) :
            predictions.append('Iris-setosa')
        elif ((sw > 2.0) & (sw <=3.0)) & ((sl > 5.0) & (sl <6.5)):
            predictions.append('Iris-versicolor')
        else:
            predictions.append('Iris-virginica')
    
    # Return the predicted list
    return pd.Series(predictions)

predictions = predictions_1(iris)




print(accuracy_score(outcomes, predictions))




sb.FacetGrid(iris, hue="Species", size=5)    .map(plt.hist, "PetalWidthCm")    .add_legend()




""" A model with few features. This model is designed based on the observation of above
    histogram. """
def predictions_2(data):
    predictions = []
    for _, species in data.iterrows():
        
        pl = float(species['PetalLengthCm'])
        if pl < 2.0:
            predictions.append('Iris-setosa')
        elif (pl >2.8) & (pl < 5.0):
            predictions.append('Iris-versicolor')
        else:
            predictions.append('Iris-virginica')
    
    # Return the predicted values
    return pd.Series(predictions)

predictions = predictions_2(iris)




print(accuracy_score(outcomes, predictions))




# Plot different combinations and observe the key factors
""" sb.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend() """
sb.set(style="whitegrid", color_codes=True)
sb.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "PetalLengthCm", "PetalWidthCm")    .add_legend()




""" A model with few features. This model is designed based on the observation of above
    scatterplot. """
def predictions_3(data):
    predictions = []
    
    for _, species in data.iterrows():
        
        pl = float(species['PetalLengthCm'])
        pw = float(species['PetalWidthCm'])
        if pl < 2.0:
            predictions.append('Iris-setosa')
        elif ((pw>0.7)&(pw<1.65)) & (pl < 5):
            predictions.append('Iris-versicolor')
        else:
            predictions.append('Iris-virginica')
    
    # Return the predicted values
    return pd.Series(predictions)

predictions = predictions_3(iris)




print(accuracy_score(outcomes, predictions))

