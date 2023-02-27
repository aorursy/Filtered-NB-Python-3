import csv as csv
import numpy as np

csv_file_object = csv.reader(open('../input/train.csv')) 
header = next(csv_file_object) 
data=[] 

for row in csv_file_object:
    data.append(row)
data = np.array(data) 




print (data) 

data[0:15,5]

(data[0::,5]) 

ages_onboard = data[0::,5].astype(np.float) 

import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('train.csv', header=0)

df

df.head(3)

(df)

df.dtypes

df.info()

df.describe()

df['Age'][0:10]

(df['Age'])

df['Age'].mean()

df[ ['Sex', 'Pclass', 'Age'] ]

df[df['Age'] > 60]

df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]

df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

for i in range(1,4):
    print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])

import pylab as P
df['Age'].hist()
P.show()




