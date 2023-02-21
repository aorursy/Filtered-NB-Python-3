#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display


data = pd.read_csv("../input/biddings.csv")
data.head()




display(data.shape)
display(data[:10])




unclicked, clicked = pd.value_counts(data['convert'].values)
total = clicked + unclicked
display((clicked / total) * 100)




get_ipython().run_line_magic('pinfo', 'np.linalg.lstsq')

