#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt




tweets = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')




tweets.head()




tweets.info()




tweets.shape




tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')




airline_sentiment = tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count()
airline_sentiment.plot(kind='bar')




features = tweets.iloc[:, 10].values
labels = tweets.iloc[:, 1].values




processed_features = []

for sentence in range(0, len(features)):
    # Remove  special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))
    # remove single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    processed_features.append(processed_feature)




from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()




is_positive = tweets['airline_sentiment'].str.contains("positive")
is_negative = tweets['airline_sentiment'].str.contains("negative")
is_neutral = tweets['airline_sentiment'].str.contains("neutral")




positive_tweets = tweets[is_positive]
positive_tweets.shape




negative_tweets = tweets[is_negative]
negative_tweets.shape




neutral_tweets = tweets[is_neutral]
neutral_tweets.shape




worst_airline = negative_tweets[['airline','airline_sentiment_confidence','negativereason']]
worst_airline   




cnt_worst_airline = worst_airline.groupby('airline', as_index=False).count()
cnt_worst_airline.sort_values('negativereason', ascending=False)




best_airline = positive_tweets[['airline','airline_sentiment_confidence']]
cnt_best_airline = best_airline.groupby('airline', as_index=False).count()
cnt_best_airline.sort_values('airline_sentiment_confidence', ascending=False)




motivation = negative_tweets[['airline','negativereason']]
cnt_bad_flight_motivation = motivation.groupby('negativereason', as_index=False).count()
cnt_bad_flight_motivation.sort_values('negativereason', ascending=False)




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)




from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=190, random_state=0)
text_classifier.fit(X_train, y_train)




predictions = text_classifier.predict(X_test)




from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))






