#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
from plotly.offline import iplot
import seaborn

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




nRowsRead = 1000 # specify 'None' if want to read whole file
df = pd.read_csv('../input/cusersmarildownloadsaspergilluscsv/aspergillus.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)
df.dataframeName = 'aspergillus.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df.head()




df.isnull().sum()




# categorical features with missing values
categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']
print(categorical_nan)




df[categorical_nan].isna().sum()




# replacing missing values in categorical features
for feature in categorical_nan:
    df[feature] = df[feature].fillna('None')




df[categorical_nan].isna().sum()




fig = px.bar(df,
             y='Serum',
             x='Protein',
             orientation='h',
             color='Count 24 Hr (of 4)Serum',
             title='Count of Aspergillus Protein in Serum',
             opacity=0.8,
             color_discrete_sequence=px.colors.diverging.Armyrose,
             template='plotly_dark'
            )
fig.update_xaxes(range=[0,35])
fig.show()




fig = px.area(df,
            x='Log2 Ratio (48 Hr/24 Hr)No_Serum',
            y='TTEST (24 Hr vs 48 Hr)No_Serum',
            template='plotly_dark',
            color_discrete_sequence=['rgb(18, 115, 117)'],
            title='',
           )

fig.update_yaxes(range=[0,2])
fig.show()




fig = px.area(df,
            x='Serum2',
            y='Count 48 Hr (of 4)Serum',
            template='plotly_dark',
            color_discrete_sequence=['rgb(18, 115, 117)'],
            title='',
           )

fig.update_yaxes(range=[0,2])
fig.show()




fig = px.bar(df, 
             x='Protein', y='Serum', color_discrete_sequence=['#27F1E7'],
             title='Count of Aspergillus fumigatus Protein in Serum', text='Count 24 Hr (of 4)Serum')
fig.show()




fig = px.bar(df, 
             x='GTEST (24 Hr vs 48 Hr)No_Serum ', y='Count 48 Hr (of 3)No_Serum', color_discrete_sequence=['crimson'],
             title='GTEST Count No-Serum 48 Hr', text='Protein')
fig.show()




fig = px.density_contour(df, x="Protein", y="Serum", title= 'Aspergillus Protein in Serum', color_discrete_sequence=['purple'])
fig.show()




fig = px.line(df, x="Log2 Ratio (48 Hr/24 Hr)No_Serum", y="Serum2", color_discrete_sequence=['darkseagreen'], 
              title="Log2 Ratio No-Serum 48 Hr")
fig.show()




fig = px.funnel_area(
    names=df['Protein'].values,
    values=df['Serum'].values,
)

fig.update_layout(
    title = 'Aspergillosis Protein in Serum'
)

fig.show()




plt.figure(figsize=(20,4))
plt.subplot(131)
sns.countplot(x= 'Serum1', data = df, palette="cool",edgecolor="black")
plt.xticks(rotation=45)
plt.subplot(132)
sns.countplot(x= 'Serum', data = df, palette="ocean",edgecolor="black")
plt.xticks(rotation=45)
plt.subplot(133)
sns.countplot(x= 'TTEST (24 Hr vs 48 Hr)No_Serum', data = df, palette="Greens_r",edgecolor="black")
plt.xticks(rotation=45)
plt.show()




ls ../input/hackathon/task_1-google_search_txt_files_v2/NL/




Netherlands = '../input/hackathon/task_1-google_search_txt_files_v2/NL/Netherlands-nl-result-13.txt'




text = open(Netherlands, 'r',encoding='utf-8',
                 errors='ignore').read()




print(text[:2000])




df1 = pd.read_csv('../input/hackathon/task_2-Tuberculosis_infection_estimates_for_2018.csv', encoding='utf8')
df1.head()




Nederland = df1[(df1['country']=='Netherlands')].reset_index(drop=True)
Nederland.head()




#word cloud
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in Nederland.country)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=200,colormap='Set2', background_color="black").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()

