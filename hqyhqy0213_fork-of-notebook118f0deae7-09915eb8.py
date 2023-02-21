#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# Some appearance options.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (10, 6)
pd.set_option('display.max_rows', 21)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/car_data.csv")
data = data.dropna()




for i in data.columns:
    data[i] = data[i].astype("category")
display(data.head())




category_col = ['make', 'fuel_type', 'aspiration', 'num_of_doors',
                'body_style', 'drive_wheels', 'engine_location', 
                'engine_type', 'num_of_cylinders','fuel_system']
numeric_col = ['wheel_base', 'length','width', 'height', 'curb_weight',
              'engine_size','compression_ratio', 'horsepower',
               'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
for i in category_col:
    data[i] = data[i].astype("category")
for i in numeric_col:
    data[i] = pd.to_numeric(data[i],errors= "coerce")
    
data = data.dropna()




for i in data.columns:
    print("%20s : %10s" %(i,data[i].dtype), ",", data.at[0,i])




# Edited formula: C(make) is included now
formula = " price ~ C(make) + C(fuel_type)+C(aspiration)+C(num_of_doors)+C(body_style)+C(drive_wheels)+C(engine_location)+wheel_base+length+width+height+curb_weight+C(engine_type)+C(num_of_cylinders)+engine_size+C(fuel_system)+compression_ratio+horsepower+peak_rpm+city_mpg+highway_mpg"
#formula = "price ~ C(fuel_system,'bbl')"
model = smf.ols(formula=formula, data = data).fit()
model.summary()




# formula with 10 features with smallest p-values. R^2 = 0.871
formula2 = " price ~ C(engine_location) + peak_rpm + curb_weight + wheel_base + width +length + engine_size + height + C(aspiration) + C(body_style)"
 
formula3 = "price ~ C(make)" # R^2 = 0.796

# smallest 10 plus "make" category. R^2 = 0.951
formula4 = " price ~ C(make) + C(engine_location) + peak_rpm + curb_weight + wheel_base + width +length + engine_size + height + C(aspiration) + C(body_style)"

formula5 = " price ~ C(make) + C(engine_location) + peak_rpm + curb_weight" # R^2 = 0.925

model = smf.ols(formula=formula4, data = data).fit()
model.summary()




predicted_price = model.predict()




predicted_price = pd.Series(predicted_price, name="PredictedPrice")




predicted_price.corr(data["price"])




data.boxplot(column="price", by="make", rot=90, grid=False)




data.corr()




plt.plot(data["wheel_base"], data["price"], 'x')
plt.xlabel("X")
plt.ylabel("Price ($)")
plt.show()




plt.hist(data["price"])
plt.xlabel("Price")
plt.ylabel("Frequency")




predicted_price = model.predict()




predicted_price = pd.Series(predicted_price, name="PredictedPrice")





len(predicted_price)
len(data["price"])




predicted_price.corr(data["price"])

