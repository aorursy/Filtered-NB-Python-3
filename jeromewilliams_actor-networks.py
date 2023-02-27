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

df = pd.read_csv('../input/movie_metadata.csv')
df.columns
df.shape

df = df[['director_name',
         'actor_1_name',
         'actor_2_name',
         'actor_3_name',
         'movie_title',
         'title_year']].copy()
df.head()

df2 = pd.get_dummies(df.head(),columns=['actor_1_name','actor_2_name','actor_3_name'])
df2.head()

df_new = df2.copy()
for column in df2.columns:
    if 'actor_1_name_' in column:
        col1_name = column
        actor_name = col1_name[len('actor_1_name_'):]
        col2_name = 'actor_2_name_' + actor_name
#         print(col2_name)
#         print(col2_name in df2.columns)
#         print(df2.columns)
        if col2_name in df2.columns:
            df_new[actor_name] = df2[col1_name] + df2[col2_name]
        else:
            df_new[actor_name] = df2[col1_name]
    if 'actor_2_name_' in column:
        col1_name = column
        actor_name = col1_name[len('actor_1_name_'):]
        
        df_new[actor_name] = df2[col1_name]

df_new.columns

df_new


