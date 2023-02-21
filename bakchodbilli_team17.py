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




indRates2014['IndividualRate'].describe()




indMean2014 = indRates2014.groupby('StateCode', as_index=False).mean()
indMean2014




for col in indMean2014.columns:
    indMean2014[col] = indMean2014[col].astype(str)
    
# set color scale
colors = [[0.0, 'rgb(242,240,247)'], [0.2, 'rgb(218,218,235)'],
         [0.4, 'rgb(188,189,220)'], [0.6, 'rgb(158,154,200)'],
         [0.8, 'rgb(117,107,177)'], [1.0, 'rgb(84,39,143)']]

indMean2014['text'] = indMean2014['StateCode'] + ' ' + 'Individuals' + ' ' + indMean2014['IndividualRate']

data = [dict(
    type = 'choropleth',
    colorscale = colors,
    autocolorscale = False,
    locations = indMean2014['StateCode'],
    z = indMean2014['IndividualRate'].astype(float),
    locationmode = 'USA-states',
    text = indMean2014['text'],
    marker = dict(
            line = dict(
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = 'Rates USD'
        )
    )]

layout = dict(
    title = '2014 US Health Insurance Marketplace Average Rates by States for Individuals',
    geo = dict(
        scope = 'usa',
    projection = dict(type='albers usa'),
    showlakes = True,
    lakecolor = 'rgb(255,255,255)',
    ),
)

fig = dict(data=data, layout=layout)

ploff.plot(data)

