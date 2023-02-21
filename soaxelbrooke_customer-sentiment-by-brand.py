#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm_notebook
import seaborn
from datetime import datetime
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")




# Load data from disk
tweets = pd.read_csv('../input/twcs/twcs.csv')

# Get customer requests and company responses 
# (from https://www.kaggle.com/soaxelbrooke/first-inbound-and-response-tweets/code)
first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]

inbounds_and_outbounds = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')

inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]

# Enable progress reporting on `df.apply` calls
tqdm_notebook().pandas()




# Instantiate sentiment analyzer from NLTK, make helper function
sentiment_analyzer = SentimentIntensityAnalyzer()

def sentiment_for(text: str) -> float:
    return sentiment_analyzer.polarity_scores(text)['compound']

sentiment_for('I love it!')




# Analyze sentiment of inbound customer support requests
inbounds_and_outbounds['inbound_sentiment'] =     inbounds_and_outbounds.text_x.progress_apply(sentiment_for)




inbounds_and_outbounds.head()




author_grouped = inbounds_and_outbounds.groupby('author_id_y')




top_support_providers = set(author_grouped.agg('count')
                                .sort_values(['tweet_id_x'], ascending=[0])
                                .index[:20]
                                .values)




inbounds_and_outbounds     .loc[inbounds_and_outbounds.author_id_y.isin(top_support_providers)]     .groupby('author_id_y')     .tweet_id_x.count()     .sort_values()     .plot('barh', title='Top 20 Brands by Volume')




inbounds_and_outbounds     .loc[inbounds_and_outbounds.author_id_y.isin(top_support_providers)]     .groupby('author_id_y')     .inbound_sentiment.mean()     .sort_values()     .plot('barh', title='Customer Sentiment by Brand (top 20 by volume)')




inbounds_and_outbounds['created_at_x'] = pd.to_datetime(inbounds_and_outbounds.created_at_x)




apple_tweets = inbounds_and_outbounds     .loc[inbounds_and_outbounds.author_id_y == 'AppleSupport']     .loc[inbounds_and_outbounds.created_at_x > datetime(2017, 10, 7)]




plt.subplot(2, 1, 1)

ax = apple_tweets     .groupby(pd.TimeGrouper(key='created_at_x', freq='24h'))     .count()     .tweet_id_x.sort_index()     .plot(title='@AppleSupport Volume & Sentiment - Impact of the iPhone X', kind='area')
ax.set_ylabel('Inbound Tweets')
    

plt.subplot(2, 1, 2)
ax = apple_tweets     .groupby(pd.TimeGrouper(key='created_at_x', freq='24h'))     .inbound_sentiment.mean()     .sort_index()     .plot(color='red')
ax.set_ylabel('Customer Sentiment')






