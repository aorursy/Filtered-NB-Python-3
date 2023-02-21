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


train = pd.read_csv( "../input/train.csv", sep= ',', encoding = 'iso-8859-1')
#product_description = pd.read_csv( "../input/product_descriptions.csv" , sep= ',', encoding = 'iso-8859-1')
attributes = pd.read_csv( "../input/attributes.csv" , sep= ',', encoding = 'iso-8859-1')

train['search_nb_token'] = train.search_term.apply( lambda x : len(x.split()) )

train

train.relevance.hist(bins=20)

train.relevance.value_counts()

(3+3+1)/3

{
3.00: [3, 3, 3],
2.33: [3, 3, 1],
2.75 : []
2.67 : [3, 3, 2],
2.50 : []
2.25 : []
2.00 : [2, 2, 2 ],
1.75 : [1,3,3]
1.67 : []
1.50 : [1,2,3]
1.33 : [1,2,2]
1.25 : [1,2,1]
1.00 : [1,1,1]

}

(1+3+3)/3

