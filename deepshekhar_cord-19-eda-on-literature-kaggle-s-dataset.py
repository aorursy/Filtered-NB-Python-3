#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from IPython.display import Image
from wordcloud import WordCloud, STOPWORDS 
from nltk.corpus import stopwords

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




import json
import os
title_list = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file = os.path.join(dirname, filename)
        if file.endswith('json'):
            with open(file) as f:
                data = json.load(f)

            #for x, y in data.items():
            #    print(x)

            for _, bib_entry in data['bib_entries'].items():
                title = bib_entry['title'].lower()
                desired_words = ['influenza','virus','pandemic']
                if all(x in title for x in desired_words):
                    #print(title)
                    if title not in title_list:
                        title_list += title.split(' ')




len(title_list)




from collections import Counter
word_could_dict = Counter(title_list)
stop_words = set(STOPWORDS).union(set(stopwords.words('english')))
#print(stop_words)
wordcloud = WordCloud(width = 1000, height = 500, stopwords = stop_words).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()




df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
df.head()
kaggle(/input/CORD-19-research-challenge/json_schema.txt)
kaggle(/input/CORD-19-research-challenge/metadata.csv)
kaggle(/input/CORD-19-research-challenge/COVID.DATA.LIC.AGMT.pdf)
kaggle(/input/CORD-19-research-challenge/metadata.readme)


























