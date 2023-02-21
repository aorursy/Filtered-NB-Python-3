#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import os
import string
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Embedding, LSTM, Dense, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import datetime
import math




def clean_imdb(directory):
    '''
    Returns cleaned dataframe of IMDB reviews with columns ['review', 'sentiment']
    '''
    sentiment = {'neg': 0, 'pos': 1}
    df_columns = ['review', 'sentiment']
    reviews_with_sentiment = pd.DataFrame(columns = df_columns)
    for i in ('test', 'train'):
        for j in ('neg', 'pos'):
            file_path = directory + i + '/' + j
            for file in os.listdir(file_path):
                with open((file_path + '/' + file), 'r',
                          encoding = 'utf-8') as text_file:
                    text = text_file.read()
                review = pd.DataFrame([[text, sentiment[j]]],
                                      columns = df_columns)
                reviews_with_sentiment = reviews_with_sentiment.                                         append(review, ignore_index = True)
    return reviews_with_sentiment

directory = '/kaggle/input/standford-imdb-review-dataset/aclimdb_v1/aclImdb/'
cleaned_imdb = clean_imdb(directory)




cleaned_imdb.iloc[13]




def load_GloVe(file_path):
    '''
    Loads word embedding .txt file
    Returns word embedding as dictionary
    '''
    GloVe_dict = dict()
    with open(file_path, encoding = 'utf-8') as GloVe_file:
        for line in GloVe_file:
            values = line.split()
            word = values[0]
            coef = np.asarray(values[1:], dtype = 'float32')
            GloVe_dict[word] = coef
    return GloVe_dict

GloVe_file_path = '../input/glove6b50d/glove.6B.50d.txt'
embedding_dict = load_GloVe(GloVe_file_path)




len(embedding_dict)




[print(word) for word in list(embedding_dict.keys()) if word != word.lower()]




list(embedding_dict.keys())[100:125]




embedding_dict['under']




cleaned_imdb.iloc[200].values




def strip_punctuation_and_whitespace(reviews_df, verbose = True):
    '''
    Strips all punctuation and whitespace from reviews EXCEPT spaces (i.e. ' ')
    Removes "<br />"
    Returns dataframe of cleaned IMDB reviews
    '''
    trans_punc = str.maketrans(string.punctuation,
                               ' ' * len(string.punctuation))
    whitespace_except_space = string.whitespace.replace(' ', '')
    trans_white = str.maketrans(whitespace_except_space,
                                ' ' * len(whitespace_except_space))
    stripped_df = pd.DataFrame(columns = ['review', 'sentiment'])
    for i, row in enumerate(reviews_df.values):
        if i % 5000 == 0 and verbose == True:
            print('Stripping review: ' + str(i) + ' of ' + str(len(reviews_df)))
        if type(reviews_df) == pd.DataFrame:
            review = row[0]
            sentiment = row[1]
        elif type(reviews_df) == pd.Series:
            review = row
            sentiment = np.NaN
        try:
            review.replace('<br />', ' ')
            for trans in [trans_punc, trans_white]:
                review = ' '.join(str(review).translate(trans).split())
            combined_df = pd.DataFrame([[review, sentiment]],
                                       columns = ['review', 'sentiment'])
            stripped_df = pd.concat([stripped_df, combined_df],
                                    ignore_index = True)
        except AttributeError:
            continue
    return stripped_df

stripped_imdb = strip_punctuation_and_whitespace(cleaned_imdb)




stripped_imdb.iloc[200].values




def get_length_all_reviews(sentences):
    '''
    Returns a list of length of all reviews
    Used for plotting histogram
    '''
    lengths = [len(i.split(' ')) for i in sentences]
    return lengths

imdb_lengths = get_length_all_reviews(stripped_imdb['review'])




max(imdb_lengths)




def plot_histogram(sentence_lengths, x_dim):
    '''
    Plots histogram of length of all sentences
    '''
    plt.hist(sentence_lengths, 50, [0, x_dim])
    plt.xlabel('Review length (words)')
    plt.ylabel('Frequency')
    plt.title('Review Lengths (Words per review)')
    plt.show()

plot_histogram(imdb_lengths, 2600)
plot_histogram(imdb_lengths, 1200)




def create_tokenizer(max_words_to_keep, words_review_df):
    '''
    Creates tokenizer
    Returns a tokenizer object and reviews converted to integers
    '''
    tokenizer = Tokenizer(num_words = max_words_to_keep,
                          lower = True,
                          split = ' ')
    tokenizer.fit_on_texts(words_review_df['review'].values)
    return tokenizer,            tokenizer.texts_to_sequences(words_review_df['review'].values)

imdb_sequence_length = 1000
vocabulary_length = 10000
tokenizer, integer_reviews = create_tokenizer(vocabulary_length, stripped_imdb)




integer_reviews[200]




print(len(integer_reviews[100]))
print(len(integer_reviews[200]))




def pad_zeros(encoded_reviews, padding_length, padding = 'pre'):
    '''
    Pads integer reviews either left ('pre') or right ('post')
    '''
    return pad_sequences(encoded_reviews,
                         maxlen = padding_length,
                         padding = padding)

padded_reviews = pad_zeros(integer_reviews,
                           imdb_sequence_length,
                           padding = 'pre')




padded_reviews[200]




split = 0.5
X_train, X_test, y_train, y_test = train_test_split(padded_reviews,
                                                    stripped_imdb['sentiment'],
                                                    test_size = split,
                                                    random_state = 42)




def create_LSTM_model(vocab_length, in_length, opt = 'Adam',
                      learning_rate = 0.001):
    '''
    Returns 1-layer LSTM model
    '''
    model = Sequential()
    model.add(Embedding(vocab_length, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation = 'sigmoid'))
    optimizer = getattr(keras.optimizers, opt)(lr = learning_rate)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizer,
                  metrics = ['accuracy'])
    return model

LSTM_model = create_LSTM_model(vocabulary_length,
                               imdb_sequence_length,
                               opt = 'Adam',
                               learning_rate = 0.001)
print(LSTM_model.summary())




ep = 10
LSTM_history = LSTM_model.fit(X_train, y_train,
                              validation_data = (X_test, y_test),
                              batch_size = 1000, epochs = ep, verbose = 1)




plt.plot(range(10), LSTM_history.history['val_acc'], '--o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy After {} Epochs'.format(ep))
plt.show()




def load_airbnb_datasets():
    '''
    Run this if you need to load in the Boston Airbnb datasets
    '''
    df_calendar = pd.read_csv('../input/boston/calendar.csv')
    df_listings = pd.read_csv('../input/boston/listings.csv')
    df_reviews = pd.read_csv('../input/boston/reviews.csv')
    return df_calendar, df_listings, df_reviews

df_calendar, df_listings, df_reviews = load_airbnb_datasets()




df_reviews.iloc[13]




df_listings.iloc[13]




ids, counts = np.unique(df_reviews['listing_id'], return_counts = True)
print('Minimum number of reviews: ' + str(min(counts)))
print('Maximum number of reviews: ' + str(max(counts)))




gt_100 = np.where(counts > 100)[0]
ids_gt_100 = ids[gt_100]

print('Number of listings with greater than 100 reviews: ' + str(len(gt_100)))
print('\nIndices of listings with greater than 100 reviews:\n' + str(gt_100))
print('\nAssociated listings with greater than 100 reviews:\n' + str(ids_gt_100))




ratings = {}

for temp_id in ids_gt_100:
    temp_comments = df_reviews.loc[df_reviews['listing_id'] ==                                    temp_id]['comments']
    
    # Rename for function, then strip punctuation and whitespace
    temp_comments.rename('review', inplace = True)
    stripped_airbnb = strip_punctuation_and_whitespace(temp_comments,
                                                       verbose = False)
    
    # Plot histogram of review length. Find sequence cutoff length
    airbnb_lengths = get_length_all_reviews(stripped_airbnb['review'])
    #plot_histogram(airbnb_lengths, 1000)
    airbnb_sequence_length = 250
    
    # Tokenizer with 10000 word vocabulary
    airbnb_tokenizer, airbnb_integer_reviews =                                             create_tokenizer(vocabulary_length,
                                                             stripped_airbnb)
    # Pad zeros up to airbnb_sequence_length
    airbnb_padded_reviews = pad_zeros(airbnb_integer_reviews,
                                      airbnb_sequence_length,
                                      padding = 'pre')
    
    # Predict sentiment
    airbnb_sentiments = LSTM_model.predict_classes(airbnb_padded_reviews)
    predicted_rating = round(airbnb_sentiments.mean() * 100, 1)

    # Print comparisons
    actual_rating = df_listings.loc[df_listings['id'] == temp_id]                    ['review_scores_rating'].values[0]
    print('--- Listing ID ' + str(temp_id) + ' ---\nPredicted Rating: [' +           str(predicted_rating) + '] vs. Actual Rating: [' +           str(actual_rating) + ']')
    ratings[temp_id] = [actual_rating, predicted_rating]




sorted_ratings = [ratings[i] for i in ratings]
sorted_ratings.sort()
sorted_ratings




plot_actual_ratings = [rating[0] for rating in sorted_ratings]
plot_predicted_ratings = [rating[1] for rating in sorted_ratings]

print('First 10 Actual Ratings: \n' + str(plot_actual_ratings[0:10]))
print('\nFirst 10 Predicted Ratings: \n' + str(plot_predicted_ratings[0:10]))




ax1_min = int(math.floor(min(plot_predicted_ratings)/5) * 5)

fig, ax1 = plt.subplots()
predicted_line = ax1.plot(range(len(plot_predicted_ratings)),
                          plot_predicted_ratings,
                          color = 'orange',
                          label = 'Predicted Ratings')
ax1.set_xlabel('Listing')
ax1.set_ylabel('LSTM Predicted Rating', color = 'orange')
ax1.tick_params(axis = 'y', color = 'orange')
plt.setp(ax1.get_yticklabels(), color = 'orange')

ax2 = ax1.twinx()
actual_line = ax2.plot(range(len(plot_actual_ratings)),
                       plot_actual_ratings,
                       color = 'black',
                       label = 'Actual Ratings')
ax2.set_ylabel('Actual Ratings', color = 'black')
ax2.set_ylim(70, 100)
ax2.spines['left'].set_color('orange')

ax1.legend((predicted_line + actual_line),
           ['Predicted Rating', 'Actual Rating'],
           loc = 'upper center',
           bbox_to_anchor = (0.5, -0.15),
           fancybox = True,
           shadow = True,
           ncol = 2)
plt.title('Predicted Ratings vs. Actual Ratings for Boston Airbnbs')
plt.show()

