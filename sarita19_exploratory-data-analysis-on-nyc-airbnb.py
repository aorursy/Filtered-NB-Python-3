#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt




#Reading the dataset.
    data = pd.read_csv(r'/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')




#Exploring the top 5 observations.
data.head()




data.describe()




data.columns.values




data.isna().sum()




total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum())/data.isnull().count().sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)
missing_data.head(40)




data.fillna({'reviews_per_month':0,'host_name': ""}, inplace=True)




data.isna().sum()




plt.figure(figsize=(10,7))
host=data['host_id'].value_counts().head(10)




viz_1=host.plot(kind='bar')
viz_1.set_xlabel('Host ID')
viz_1.set_ylabel('Count of Listings')
viz_1.set_title("Listing of AirBnb homes for the various hosts")
viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)




data.neighbourhood_group.value_counts()




sub_data = data[data.price < 500]
plot_2=sns.violinplot(data=sub_data, x='neighbourhood_group', y='price')
plot_2.set_title('Density and distribution of prices for each neighberhood_group')




sub_data = data[data.price < 500]
plt.figure(figsize = (12, 6))
sns.boxplot(x = 'room_type', y = 'price',  data = sub_data)




data['neighbourhood'].value_counts()




data['neighbourhood'].describe()




data.neighbourhood.nunique()




data.groupby(['neighbourhood']).mean()




data.sort_values(by='price', inplace=True, ascending=False)




data.price.describe()




top10_freq_nghrbhd=data.neighbourhood.value_counts().head(10)
top10_freq_nghrbhd_data=data[data['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',
                 'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown'])]
top10_freq_nghrbhd_data




sns.catplot(x="neighbourhood", y="price", col="room_type", data=top10_freq_nghrbhd_data)






