#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ggplot import * # plotting library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




data = pd.read_csv('../input/HR_comma_sep.csv')




data.describe()




basicplot = ggplot(aes(x='satisfaction_level', y='left'), data=data) +    stat_smooth()
    
print(basicplot)




satxtimespend = ggplot(aes(x='satisfaction_level', color = 'time_spend_company'), data=data) +    geom_line()
    
print(satxtimespend)
