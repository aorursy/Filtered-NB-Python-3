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




debate = pd.read_csv('../input/debate.csv')
debate.head(10)




debate = debate.loc[debate.Date == "2016-09-26"]
debate.Speaker.drop_duplicates()
debate[(debate.Speaker == "CANDIDATES") & (debate.Text != "(CROSSTALK)")]
debate[debate['Text'].str.contains("Wrong")]
[x.count("China") for x in debate[debate['Text'].str.contains("China")].Text




trumpWords = " ".join(debate.loc[debate.Speaker == "Trump"]["Text"])
clintonWords = " ".join(debate.loc[debate.Speaker == "Clinton"]["Text"])

wordcloud = WordCloud().generate(text = trumpWords)

plt.imshow(wordcloud)
plt.axis("off")

# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(trumpWords)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

wordcloud = WordCloud().generate(text = clintonWords)

plt.imshow(wordcloud)
plt.axis("off")

# take relative word frequencies into account, lower max_font_size
wordcloud = WordCloud(max_font_size=30, relative_scaling=.5).generate(clintonWords)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

