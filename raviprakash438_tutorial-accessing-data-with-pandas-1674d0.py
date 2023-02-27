#!/usr/bin/env python
# coding: utf-8



import pandas as pd




df = pd.read_csv('../input/parks.csv', index_col=['Park Code'])




df.head(3)




df.info()



df.loc['BADL']




df.loc[['BADL', 'ARCH', 'ACAD']]




df[:3]




df[2:6]




df['State'].head(3)




df.State.head(3)



df.columns = [col.replace(' ', '_').lower() for col in df.columns]
print(df.columns)




df[['state', 'acres']][:3]




(df.state == 'UT').head(3)




df[df.state == 'UT']



df[df['park_name'].str.split().apply(lambda x: len(x) == 3)].head(3)




df[df.state.isin(['WA', 'OR', 'CA'])].head()
