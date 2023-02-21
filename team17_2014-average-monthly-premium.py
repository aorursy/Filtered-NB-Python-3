#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




import plotly.plotly as py
import plotly.offline as ploff

from ggplot import *
from subprocess import check_output
from plotly.offline import init_notebook_mode, plot

ploff.init_notebook_mode()

headers = ['BusinessYear', 'StateCode', 'Age', 
           'IndividualRate', 'Couple']

# read in chuncks for memory efficiency
filePath = '../input/Rate.csv'
chunks = pd.read_csv(filePath, iterator=True, chunksize=1000,
                    usecols=headers)
rates = pd.concat(chunk for chunk in chunks)

randomRows = rates.sample(n=6)
randomRows




pd.set_option('display.float_format', lambda x: '%.2f' % x)
print (rates['Couple'].describe())




print (rates['IndividualRate'].describe())




ratesInd9000 = rates[rates.IndividualRate < 9000]
print (ratesInd9000['IndividualRate'].describe())




graph1 = ggplot(aes(x='Couple'), data=rates) +     geom_histogram(binwidth=10) +     ggtitle('Distribution of Couple Rates')
    
print (graph1)




graph2 = ggplot(aes(x='IndividualRate'), data=ratesInd9000) +     geom_histogram(binwidth=25, colour='red') +     ggtitle('Distribution of Individual Rates')
    
print (graph2)




indRate1200 = ratesInd9000[ratesInd9000.IndividualRate > 1200].count()['IndividualRate']
percentageOfTotalInd9000 = indRate1200 / ratesInd9000['IndividualRate'].describe()['count']
print ('%i individual plans have a rate greater than $1200. Thats %% %f of the total number of IndividualRate plans that we filtered out below $9000' % (indRate1200, percentageOfTotalInd9000))




columns = ['BusinessYear', 'StateCode', 'IndividualRate']
indRates = pd.DataFrame(ratesInd9000, columns=columns)
indRates2014 = indRates[indRates.BusinessYear == 2014]
indRates2014 = indRates2014.dropna(subset=['IndividualRate'])
randomRows2014 = indRates2014.sample(n=6)
randomRows2014




indMean2014 = indRates2014.groupby('StateCode', as_index=False).mean()
indMean2014

