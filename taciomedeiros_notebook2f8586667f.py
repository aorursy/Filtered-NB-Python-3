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




f = open("../input/HR_comma_sep.csv")




l = f.readline()




print(l)




header = l.split(',')




print(header)



header = l[:-1].split(',')




print(header)




f.seek(0)




l.readline()




f.readline()
f.readline()




f.seek(0)




dados = [x[:-1].split(',') for ]

