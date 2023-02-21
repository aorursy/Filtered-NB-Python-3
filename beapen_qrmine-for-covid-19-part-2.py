#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.head(5)




df_cleaned = pd.concat([df['Patient ID'], df['Patient age quantile'], df['Eosinophils'], df['Leukocytes'], df['Hematocrit'], df['Hemoglobin'], df['Platelets'], df['SARS-Cov-2 exam result']], axis = 1)
df_cleaned.head(5)




# rename columns
df_cleaned.columns = ['id', 'age', 'eos', 'leu', 'hct', 'hb', 'pt', 'covid']
df_cleaned.head(5)




# encode DV
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_cleaned2 = df_cleaned
# Fit the encoder to the pandas column\n",
le.fit(df_cleaned2['covid'])
df_cleaned2['covid'] = le.transform(df_cleaned2['covid'])
df_cleaned2.head(5)




df_cleaned2.fillna(df_cleaned2.mean(), inplace=True)
#df_cleaned2.dtypes
df_cleaned2['covid'] = df_cleaned2['covid'].astype('bool')
df_cleaned2.to_csv('/tmp/qrmine-quan.csv', index = False)
df_cleaned2.head(5)




get_ipython().system('pip install qrmine==3.4.0')




get_ipython().system('pip install scikit-learn==0.22.0')




from qrmine import MLQRMine




ml = MLQRMine()
ml.csvfile = "/tmp/qrmine-quan.csv"
ml.read_csv()
ml.epochs = 5
ml.prepare_data(True)  # Oversample
ml.get_nnet_predictions()
print("\n%s: %.2f%%" % (ml.model.metrics_names[1], ml.get_nnet_scores()[1] * 100))




knn = ml.knn_search(3, 3)
for n in knn:
    print(n)




ml.get_pca(3)




ml.prepare_data()
ml.get_kmeans(3)

