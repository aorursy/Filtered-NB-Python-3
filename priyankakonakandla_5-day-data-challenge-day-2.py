#!/usr/bin/env python
# coding: utf-8



import pandas as pd 




data = pd.read_csv("../input/cereal.csv")




data.describe()




import matplotlib.pyplot as plt




protein = data['protein']
print(protein)




plt.hist(protein, bins=6, color="pink",edgecolor="black")
plt.title("Hist plot")




plt.hist(column = "protein")

