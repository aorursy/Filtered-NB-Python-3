#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




train = pd.read_csv('../input/Train_A102.csv')




train.head(100)




train.shape




train.max()




train.min()




train.isnull().sum()




train.Outlet_Size.describe()




train.Item_Weight.describe()




train.boxplot(column='Item_Weight')
plt.show()




train.boxplot(column='Item_Outlet_Sales')
plt.show()




train.notnull().head()




train.Item_Outlet_Sales.describe()




new_train = train.dropna(how='any')




new_train.head()




train.Outlet_Size.value_counts(dropna=False)




new_train.Outlet_Size.value_counts(dropna=False)




train.Item_Weight.value_counts(dropna=False)




new_train.boxplot(column='Item_Weight')
plt.show()




new_train.boxplot(column='Item_Outlet_Sales')
plt.show()




train_plot = train.head(50)
plt.scatter(train_plot.Item_Visibility,train_plot.Item_Outlet_Sales,color='orange')
plt.xlabel('Visibility')
plt.ylabel('Sales')
plt.title('Sales vs Visibility')
plt.show()




train_plot_2 = train.tail(50)
plt.scatter(train_plot_2.Item_Visibility,train_plot_2.Item_Outlet_Sales,color='orange')
plt.xlabel('Visibility')
plt.ylabel('Sales')
plt.title('Sales vs Visibility')
plt.show()




train_plot_3 = train.head(50)
plt.scatter(train_plot_3.Item_MRP,train_plot_3.Item_Outlet_Sales,color='green')
plt.xlabel('MRP')
plt.ylabel('Sales')
plt.title('Sales vs MRP')
plt.show()




train_plot_4 = train.tail(50)
plt.scatter(train_plot_4.Item_MRP,train_plot_4.Item_Outlet_Sales,color='green')
plt.xlabel('MRP')
plt.ylabel('Sales')
plt.title('Sales vs MRP')
plt.show()




train_plot_5 = train.head(50)
fig=plt.figure()
graph1=fig.add_subplot(1,2,1)
graph1.scatter(train_plot_5.Item_Visibility,train_plot_5.Item_Outlet_Sales)
graph1.set_title("Sales vs Visibility")
graph2=fig.add_subplot(1,2,2)
graph2.scatter(train_plot_5.Item_MRP,train_plot_5.Item_Outlet_Sales,color="green")
graph2.set_title("Sales vs MRP")
plt.show()




new_train_1=train.fillna(method="ffill")
new_train_1.head(100)#forward fill method




new_train_1.boxplot(column='Item_Weight')
plt.show()




new_train_2=train.fillna(method="bfill")
new_train_2.head(100)#backward fill method




new_train_2.boxplot(column='Item_Weight')
plt.show()




new_train_3 = train.interpolate()
new_train_3.head(100)




new_train_3.boxplot(column='Item_Weight')
plt.show()




new_train_3.Item_Weight.describe()




MRP_Type = []
for mrp in train.Item_MRP:
   if mrp < 70:
        MRP_Type.append('Low')
   elif mrp < 130:
        MRP_Type.append('Medium')
   elif mrp < 201:
        MRP_Type.append('High')
   else:
        MRP_Type.append('Very High')
train['Item_MRP_Types'] = MRP_Type
train.head()




new_train_4 = train.fillna(train.mean())




new_train_4.head(50)




new_train_4.boxplot(column='Item_Weight')
plt.show()




new_train_5 = train.fillna(train.median())
new_train_5.head(50)




new_train_5.boxplot(column='Item_Weight')
plt.show()




train.boxplot(return_type='dict',rot=90)
plt.plot()
plt.show()




train.Item_Outlet_Sales.describe()




train.Item_Visibility.describe()




item_visibility_outliers = train[train.Item_Visibility > 0.195979]
item_visibility_outliers.head()




item_outlet_sales_outliers=train[train.Item_Outlet_Sales > 6501.8699]
item_outlet_sales_outliers.head()




train_removed_outliers_1 = train[train.Item_Outlet_Sales < 6501.8699]
train_removed_outliers_1.head(100)




train_removed_outliers_2 = train_removed_outliers_1[train_removed_outliers_1.Item_Visibility < 0.195979]
train_removed_outliers_2.head(100)




train_removed_outliers_2.shape




train_removed_outliers_2.Item_Outlet_Sales.describe()




train_removed_outliers_2.boxplot(return_type='dict',rot=90)
plt.plot()
plt.show()






