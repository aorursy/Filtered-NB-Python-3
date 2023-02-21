#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




df = pd.read_csv('../input/inpatientCharges.csv')




df.head(3)




df.describe()




df.columns.values




df['Provider Name'].unique




df['AMP'] = df['Average Medicare Payments'].str.lstrip('$').astype('float')
av_medicare_plot = plt.hist(df['AMP'], bins = 200)




df['ATP'] = df[' Average Total Payments '].str.lstrip('$').astype('float')
av_total_plot = plt.hist(df['ATP'], bins = 200)




df['ACC'] = df[' Average Covered Charges '].str.lstrip('$').astype('float')
av_total_plot = plt.hist(df['ACC'], bins = 200)




plt.scatter(x = df[' Total Discharges '], y = df.ACC)




Back to Work... To Be Continued

