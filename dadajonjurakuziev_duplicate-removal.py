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




import pandas as pd

df = pd.read_csv("/kaggle/input/movieposter/labeled_50K.csv")




test = df[df.duplicated(['title', 'poster'])]
duplicated_titles_and_url = list(test['title'])




duplicated_id = list()
for i in range(len(df['title'])):
    for j in range(len(duplicated_titles_and_url)):
        if df['title'][i] == duplicated_titles_and_url[j]:
            print(f"{duplicated_titles_and_url[j]} : {i}")
            duplicated_id.append(i)




idx = [558, 5661, 10058, 11474, 14261, 34535]
for i in idx:
    duplicated_id.remove(i)
    
for idx in duplicated_id:
    df = df[df['id'] != idx]
df.shape




duplicates = []
titles = list(df.title)

for i in range(len(titles)):
    if "deleted scenes" in titles[i].lower() or "alternate endings" in titles[i].lower()     or "alternate opening" in titles[i].lower() or "director's cut" in titles[i].lower():
        duplicates.append(titles[i])




for dupl in duplicates:
    df = df[df['title'] != dupl]
df.shape




gemini_dupl = [18909, 19575, 28592, 23929, 24426, 10111, 33566, 31229, 31360, 31460, 31899, 
               14128, 5841, 40085, 11077, 31206, 17855, 21567, 2975, 16712, 25937, 31284, 
               31253, 605, 5444, 36893, 38051, 7001, 11542, 34584, 2225, 2248]

for dupl in gemini_dupl:
    df = df[df['id'] != dupl]
df.shape




posters_to_delete = []

for i in range(len(df['title'])):
    if i in df.id:
        continue
    else:
        posters_to_delete.append(i)
posters_to_delete




df.head()




stats = {'drama': 16318,
         'comedy': 14410,
         'action': 10537,
         'romance': 7432,
         'horror': 7291,
         'thriller': 7099,
         'crime': 6119,
         'adventure': 5693,
         'animation': 5318,
         'mystery': 4760,
         'short': 4155,
         'fantasy': 3819,
         'sci-fi': 3609}




import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

x = stats.keys()
y = stats.values()

x_pos = [i for i, _ in enumerate(x)]

plt.rcParams["figure.figsize"] = (14,4)
plt.bar(x_pos, y, color='g')
plt.xlabel("Labels")
plt.ylabel("Number of samples")
plt.title("41K movie posters")
plt.xticks(x_pos, x)

plt.show()






