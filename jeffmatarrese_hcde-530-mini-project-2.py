#!/usr/bin/env python
# coding: utf-8



# importing dependency libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize
import  nltk,json, math, time, requests

# API keys
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("monkey_learn")
secret_value_1 = user_secrets.get_secret("newsapi")




outlets = ['abc-news',
           'cnn',
           'the-huffington-post',
           'fox-news',
           'usa-today',
           'reuters',
           'politico',
           'the-washington-post',
           'nbc-news',
           'cbs-news',
           'newsweek',
           'new-york-magazine',
           'rt',
           'the-hill',]




def getStories(query,data):
    key = secret_value_1
    for outlet in outlets:    #loop through outlets
        df_loop = pd.DataFrame()     #empty dataframe to collect page data
        outlet_data = requests.get("http://newsapi.org/v2/everything?qInTitle="+query+"&sources="+outlet+"&pageSize=100&apiKey="+key)     #save the request
        df_loop = json_normalize(outlet_data.json()['articles'])     #add json to empty frame
        data = pd.concat([data, df_loop], ignore_index=True)     #concat to external dataframe
    return data




biden_df = pd.DataFrame()
biden_df = getStories('biden',biden_df)

# clean up the dataframe for viewing. Don't need image URLs, timestamps, etc.
def clean_dataFrame(dataframe, candidate):
    dataframe = dataframe[['title','source.name']]    #only take title and sorucename.
    dataframe = dataframe.dropna()     #remove any n/a results
    dataframe['candidate'] = candidate    #add column for candidate name in case we expand this to multiple candidates
    dataframe = dataframe.reset_index(drop=True)    #reset the index if any rows were dropped.
    return dataframe

clean_dataFrame(biden_df, 'Biden')




biden_train = biden_df.sample(frac=0.8,random_state=1)
biden_test = biden_df.drop(biden_train.index)

# biden_train.to_excel('biden_train.xlsx')
# biden_test.to_excel('biden_test.xlsx')




biden_test = pd.read_excel('/kaggle/input/hcde530-mp2-data/biden_test.xlsx') #loading data into pd dataframe
biden_test = biden_test[['title','source.name']]
biden_test.tail()




# installing and importing the MonkeyLearn modules
get_ipython().system('pip install monkeylearn')
from monkeylearn import MonkeyLearn

# Method for accessing ML model
def monkey_analyze(headline):
    ml = MonkeyLearn(secret_value_0)
    data = [str(headline)]
    model_id = 'cl_EvQHu6MT'
    result = ml.classifiers.classify(model_id, data)
    return result.body[0]['classifications'][0]['tag_name']




# Adding vader sentiment module
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()

# apply analysis and score results. 0.05 threshold currently being used.
def vader_analyze(headline):
    snt = sia.polarity_scores(headline)
    if snt['compound'] >= 0.05:
        return "Positive"
    elif snt['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"




def sent_analysis(data):
    data['vader sentiment'] = data['title'].apply(lambda headline: vader_analyze(headline))
    data['monkey sentiment'] = data['title'].apply(lambda headline: monkey_analyze(headline))




#sent_analysis(biden_test)
#biden_test.to_excel('biden_test_results.xlsx')




fndgs = pd.read_excel('/kaggle/input/hcde530-mp2-data/biden_test_results.xlsx')




import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

vader = px.histogram(fndgs, x='vader sentiment', 
                             color='vader sentiment', 
                             title='Sentiment from VADER analysis')
vader.show()




monkey = px.histogram(fndgs, x='monkey sentiment', 
                             color='monkey sentiment', 
                             title='Sentiment from MonkeyLearn model')
monkey.show()

