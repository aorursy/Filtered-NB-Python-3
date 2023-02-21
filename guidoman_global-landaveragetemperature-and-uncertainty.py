#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="white", color_codes=True)

from scipy import stats




data_dir = '../input/'




global_t = pd.read_csv(data_dir + 'GlobalTemperatures.csv')
global_t.index = pd.to_datetime(global_t['dt'],infer_datetime_format=True)
del global_t['dt']
global_t.tail()




global_by_year = global_t.groupby(global_t.index.year)
avg_by_year = global_by_year.mean()['LandAverageTemperature']




f, ax = plt.subplots(figsize=(11, 6))
_ = avg_by_year.plot(ax=ax)




f, ax = plt.subplots(figsize=(11, 6))
temp_uncertainty = global_t['LandAverageTemperatureUncertainty'].dropna()
_ = temp_uncertainty.plot(ax=ax)




from scipy.stats import skew
skewness = skew(temp_uncertainty)
print('Skewness of "LandAverageTemperatureUncertainty" is {}'.format(skewness))




avg_uncertainty_by_year = global_t.groupby(global_t.index.year)     .mean()['LandAverageTemperatureUncertainty']
avg_uncertainty_smaller_than_median = avg_uncertainty_by_year[
    avg_uncertainty_by_year < avg_uncertainty_by_year.median()]
print('The median value of "LandAverageTemperatureUncertainty" means by year is {:.4f}'       .format(avg_uncertainty_by_year.median()))




prev_year_inc = (avg_by_year.shift(1) - avg_by_year).dropna()

f, ax = plt.subplots(figsize=(11, 6))
_ = prev_year_inc.plot(ax=ax)




f, ax = plt.subplots(figsize=(11, 6))
_ = prev_year_inc.ix[avg_uncertainty_smaller_than_median.index].plot(ax=ax)




reliable_avg_temps = avg_by_year[avg_uncertainty_smaller_than_median.index]




import numpy as np
from scipy.signal import argrelextrema




maxima_idx = argrelextrema(reliable_avg_temps.as_matrix(), np.greater)
maxima = reliable_avg_temps.iloc[maxima_idx[0]]

# one more time: compute the local maxima of the local maxima
maxima2_idx = argrelextrema(maxima.as_matrix(), np.greater)
# the numeric indexes in "maxima2_idx" refer to the "maxima" series.
maxima2  = reliable_avg_temps.ix[maxima.index[maxima2_idx]]
print('Maxima maxima:')
print(maxima2)




f, ax = plt.subplots(figsize=(11, 6))
reliable_avg_temps.plot(ax=ax)
plt.scatter(maxima.index, maxima.as_matrix(), color='r')
plt.scatter(maxima2.index, maxima2.as_matrix(), color='k')
pass




import scipy
first_half_XX = reliable_avg_temps.ix[1901:1950]
second_half_XX = reliable_avg_temps.ix[1951:2000]
ttest = scipy.stats.ttest_ind(first_half_XX, second_half_XX)
print('T-test p-value = {:.20f}'.format(ttest[1]))

