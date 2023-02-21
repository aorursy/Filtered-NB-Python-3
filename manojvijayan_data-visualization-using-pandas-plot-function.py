#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




df = pd.read_csv('../input/anonymous-survey-responses.csv')




df.info()




df.describe().transpose()




cols = ['T_C_S', 'P_E_P', 'I_D_S', 'P_D_C']




m_df = df.copy()




m_df.columns = cols




m_df.head(1)




m_df.T_C_S.value_counts().plot(kind='bar')




m_df.P_E_P.value_counts().plot(kind='bar')




m_df.P_E_P.replace(to_replace='I have a whole lot of expereince', value='I have a whole lot of experience', 
                   inplace=True)




m_df.P_D_C.value_counts()




m_df.P_D_C = m_df.P_D_C.apply(lambda val: str(val).split(' ')[0])




plt.figure(figsize=(10,10))
ax1 = m_df.P_E_P.value_counts().plot(kind='bar',color='grey',alpha=0.4)
ax1.set_yticks(np.arange(0, 550, 50.0))
m_df[(m_df.T_C_S == "Yep")]         .P_E_P.value_counts().plot(kind='bar',color='green',alpha=0.6)
m_df[(m_df.T_C_S == "Yes, but I've forgotten everything")].P_E_P.value_counts().plot(kind='bar',color='Orange',alpha=0.6)
m_df[(m_df.T_C_S == "Nope")].P_E_P.value_counts().plot(kind='bar',color='red',alpha=0.6)




plt.figure(figsize=(10,10))
ax1 = m_df.P_E_P.value_counts().plot(kind='bar',color='grey',alpha=0.4)
ax1.set_yticks(np.arange(0, 550, 50.0))
m_df[(m_df.I_D_S == "I want to get a job where I use data science")]         .P_E_P.value_counts().plot(kind='bar',color='green',alpha=0.6)
m_df[(m_df.I_D_S == "It will help me in my current job")].P_E_P.value_counts().plot(kind='bar',color='Orange',alpha=0.6)
m_df[(m_df.I_D_S == "Just curious")].P_E_P.value_counts().plot(kind='bar',color='red',alpha=0.6)
m_df[(m_df.I_D_S == "other")].P_E_P.value_counts().plot(kind='bar',color='blue',alpha=0.6)




plt.figure(figsize=(10,10))
ax1 = m_df.P_E_P.value_counts().plot(kind='bar',color='grey',alpha=0.4)
ax1.set_yticks(np.arange(0, 550, 50.0))
m_df[(m_df.P_D_C == "Dogs")]         .P_E_P.value_counts().plot(kind='bar',color='green',alpha=0.6)
m_df[(m_df.P_D_C == "Both")].P_E_P.value_counts().plot(kind='bar',color='blue',alpha=0.6)
m_df[(m_df.P_D_C == "Cats")].P_E_P.value_counts().plot(kind='bar',color='Orange',alpha=0.6)
m_df[(m_df.P_D_C == "Neither")].P_E_P.value_counts().plot(kind='bar',color='red',alpha=0.6)











