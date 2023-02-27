#!/usr/bin/env python
# coding: utf-8



# Import stuff.




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualisation
import matplotlib.pyplot as plt # plot visualisation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output




train = pd.read_csv(os.path.join('../input', 'train.csv'))
test = pd.read_csv(os.path.join('../input', 'test.csv'))




train['Fare'] = [round(elem) for elem in train['Fare']]
sns.swarmplot(x=train['Fare'], y=train['Survived'])




sns.countplot(train['Pclass'], hue=train['Survived'])

