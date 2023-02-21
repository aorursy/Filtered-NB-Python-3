#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




data=pd.read_csv('../input/haberman.csv/haberman.csv')
print(data.shape)
data['status']=data['status'].map({1:'survived',2:'not survived'}) #mapping status




data.head()




data.info()




sns.FacetGrid(data,hue='status',height=5)   .map(sns.distplot,'age')   .add_legend()
plt.show()




sns.FacetGrid(data,hue='status',height=5)   .map(sns.distplot,'year')   .add_legend()
plt.show()




sns.FacetGrid(data,hue='status',height=5)   .map(sns.distplot,'nodes')   .add_legend()
plt.show()




sns.set_style('whitegrid')
sns.pairplot(data, hue='status')
plt.show()






