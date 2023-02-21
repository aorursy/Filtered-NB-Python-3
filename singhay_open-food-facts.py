#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




# Make plots larger
plt.rcParams['figure.figsize'] = (15, 9)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"




data = pd.read_csv(filepath_or_buffer='../input/en.openfoodfacts.org.products.tsv', delimiter="\t")




data.head()




data.shape




data.loc[data['cities_tags'].notnull(), ['product_name', 'cities', 'cities_tags']]




data.loc[data['labels'].notnull(), ['product_name', 'labels', 'labels_n', 'labels_tags']]




# data.loc[data['labels_tags'].notnull(), ['product_name', 'labels', 'labels_tags']].shape
# data.loc[:,['stores', 'countries', 'countries_en', 'countries_tags']]
data.loc[data['additives'].notnull(), ['additives_n', 'additives', 'additives_en', 'additives_tags']]




# Deleting this since it's all NaN
data[data['no_nutriments'].notnull() == True].shape




data[data['ingredients_text'].notnull() == True].shape




list(data.columns.values)




# Drop columns who have a better representation of themselves e.g. cities_tags is better than cities
data.drop([  'url',
             'emb_codes', 
             'labels',
             'first_packaging_code_geo',
             'cities',
             'countries',
             'countries_en',
             'traces',
             'packaging',
             'brands',
             'categories',
             'categories_en',
             'origins',
             'manufacturing_places',
             'allergens',
             'traces_tags',
             'additives',
             'additives_tags',
             'additives_en',
             'no_nutriments',
             'image_url',
             'image_small_url', axis=1, inplace=True)




data.shape




data.describe()




# these columns have no values i.e. unique value count should be > 0
drop_list = []
for cname in data.columns:
    if data[cname].nunique() == 0:
        print(cname)
        drop_list.append(cname)




data.drop(drop_list, axis=1, inplace=True)




data.shape




data.info()




# One can not really find the true count since multiple countries are make a particular product too
data['countries_tags'].value_counts()[:20]




data_macro = data.loc[:, ['fat_100g',
                         'monounsaturated-fat_100g',
                         'polyunsaturated-fat_100g', 
                         'trans-fat_100g',
                         'carbohydrates_100g',
                         'sugars_100g',
                         'fiber_100g',
                         'proteins_100g']]




data_macro.info()




data_macro.fillna(0.0)




data_macro['monounsaturated-fat_100g'].value_counts().shape
data_macro['polyunsaturated-fat_100g'].value_counts().shape
data_macro['trans-fat_100g'].value_counts().shape




# pd.isnull(data_macro['fat_100g']).shape
sns.distplot(data_macro['fat_100g'])






