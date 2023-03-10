#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head(5)

df.groupby('Sex').size()/len(df)*100

df['Cabin'].unique()

len(df[df["Parch"] >= 2])


# highest value fare

df[df["Fare"] == max(df["Fare"])]["Embarked"].unique()[0]

df.groupby("Embarked")["Fare"].mean()

df[df['Age'] >= 50].groupby('Cabin').size().sort_values(ascending=False)

df[['Pclass', 'Embarked', 'Sex', 'Survived']].groupby(['Pclass', 'Embarked', 'Sex']).agg(['count', 'sum', 'mean'])