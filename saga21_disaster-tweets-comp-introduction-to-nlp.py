#!/usr/bin/env python
# coding: utf-8



# General libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns

# Libraries for data cleaning
import unidecode
import re
import string
get_ipython().system('pip install pycontractions')
from pycontractions import Contractions
import gensim.downloader as api
# Choose model accordingly for contractions function
model = api.load("glove-twitter-25")
# model = api.load("glove-twitter-100")
# model = api.load("word2vec-google-news-300")
cont = Contractions(kv_model=model)
cont.load_models()
import operator

# NLP libraries
import spacy
from spacy.lang.en import English
get_ipython().system('pip install pyspellchecker')
from spellchecker import SpellChecker
import nltk
from nltk.corpus import stopwords

# ML libraries
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# NN libraries
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint




# List of files (including output files)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
test = pd.read_csv("../input/nlp-getting-started/test.csv")
train = pd.read_csv("../input/nlp-getting-started/train.csv")

print("Train dataset size: ", len(train))
print("Test dataset size: ", len(test))

# A brief look on the train dataset
train[90:100]




# Join  train and test datasets to analyze all data in single commands
all_data = pd.concat([train,test], axis = 0, sort=False)

# How many locations/keywords?
print("Number of unique locations: ", all_data.location.nunique())
print("Number of unique keywords: ",all_data.keyword.nunique())

# Check data balance; frequency of real disaster tweets (target=1)
print("Percentage of real disaster tweets: ", all_data.target.sum()/len(train)*100)

# Find missing values in a single code line,
print("Missing values:")
display(all_data.isna().sum())

ax = sns.barplot(train['target'].value_counts().index,train['target'].value_counts()/len(train))
ax.set_title("Real vs fake tweets")
ax.set_ylabel("Percentage")
ax.set_xlabel("Target")




f1, axes = plt.subplots(1, 2, figsize=(15,5))
f1.subplots_adjust(hspace=0.4, wspace=0.3)
ax1 = sns.barplot(y=train['keyword'].value_counts()[:15].index,x=train['keyword'].value_counts()[:15],
            orient='h', ax=axes[0])
ax1.set_title("Top 15 keywords")
ax2 = sns.barplot(y=train['location'].value_counts()[:15].index,x=train['location'].value_counts()[:15],
            orient='h', ax=axes[1])
ax2.set_title("Top 15 locations")




# Fill missings for train and test datasets
all_data.location.fillna("None", inplace=True)
all_data.keyword.fillna("None", inplace=True)

# Fill the target column for the test rows. This will help us with future calls to model predictions
all_data.target.fillna(0, inplace=True)




# Define the vectorizer counter that will compute the BOW
def count_vector(data):
    count_vectorizer = CountVectorizer()
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer


# Define a function for the logistic regression algorithm
def logreg_bow(data, valid_fraction):
    
    # Transform data to list
    list_corpus = data["text"].tolist()
    list_labels = data["target"].tolist()

    # Split train-validation data
    X_train, X_valid, y_train, y_valid = train_test_split(list_corpus, list_labels, 
                                                          test_size=valid_fraction, random_state=21)

    # Generate the bag of words through the count_vectorizer function
    X_train_counts, count_vectorizer = count_vector(X_train)
    X_valid_counts = count_vectorizer.transform(X_valid)

    # Run LogisticRegression model
    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                             multi_class='multinomial', n_jobs=-1, random_state=21)
    clf.fit(X_train_counts, y_train)
    y_predicted_counts = clf.predict(X_valid_counts)
    
    # Cross validation score over 10 folds
    scores = cross_val_score(clf, X_train_counts, y_train, cv=10)
    print("Cross validation over 10 folds: ", scores, " --- ", sum(scores)/10.)

    return y_predicted_counts, y_valid


# Define a metrics function named get_metrics to evaluate the model's performance
def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


# Final call of the logistic regression model and metrics
y_predicted_logistic, y_val = logreg_bow(all_data[:len(train)], 0.2)
accuracy, precision, recall, f1 = get_metrics(y_val, y_predicted_logistic)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))




def submit_logistic(outfile_name, data, test, valid_frac):
    
    # Run the LogisticRegression model
    y_pred, y_valid = logreg_bow(data, valid_frac)
    
    # Submit results
    submission = pd.DataFrame({
            "id": test.id, 
            "target": y_pred
        })
    submission.to_csv(outfile_name, index=False)
    
    return y_pred, y_valid
    
y_logistic_pred, y_logistic_test = submit_logistic("submission_logistic_basic.csv", all_data, test, len(test)/len(all_data))




# Convert text column to lowercase
all_data2 = all_data.copy()
all_data2['text'] = all_data['text'].apply(lambda x: x.lower())
all_data2.head(5)
all_data2.to_csv("tweets_lowercased", index=False)




y_logistic_lower, y_val = logreg_bow(all_data2[:len(train)], 0.2)
accuracy, precision, recall, f1 = get_metrics(y_val, y_logistic_lower)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))




def remove_accented_chars(text):
    text = unidecode.unidecode(text)
    return text

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

def expand_contractions(text):
    text = list(cont.expand_texts([text], precise=True))[0]
    return text

spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

# Now compact all the normalization function calls into a single function
def normalization(text):
    text = remove_accented_chars(text)
    text = remove_html(text)
    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_punct(text)
    text = expand_contractions(text)
    text = correct_spellings(text)
    return text




ts = time.time()

all_data2 = all_data.copy()
all_data2['text'] = all_data['text'].apply(lambda x: normalization(x))
all_data2.head(5)
all_data2.to_csv("tweets_normalized", index=False)

print("Time spent: ", time.time() - ts)




y_logistic_norm, y_val = logreg_bow(pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_normalized")[:len(train)], 0.2)
accuracy, precision, recall, f1 = get_metrics(y_val, y_logistic_norm)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))




# Load the spacy model to get sentence vectors
nlp = spacy.load('en_core_web_lg')




def remove_stopwords(text):
    tokens = [token.text for token in text if not token.is_stop]
    return ' '.join([token for token in tokens])

ts = time.time()
all_data2 = all_data.copy()
all_data2['text'] = all_data['text'].apply(lambda x: remove_stopwords(nlp(x)))
all_data2.head(5)
all_data2.to_csv("tweets_no_stopwords", index=False)
print("Time spent: ", time.time() - ts)




y_logistic_stop, y_val = logreg_bow(pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_no_stopwords")[:len(train)], 0.2)
accuracy, precision, recall, f1 = get_metrics(y_val, y_logistic_stop)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))




def lemmatizer(text):
    tokens = [token.lemma_ for token in text]
    return ' '.join([token for token in tokens])

ts = time.time()
all_data2 = all_data.copy()
all_data2['text'] = all_data['text'].apply(lambda x: lemmatizer(nlp(x)))
all_data2.head(5)
all_data2.to_csv("tweets_lemmatized", index=False)
print("Time spent: ", time.time() - ts)




y_logistic_lemma, y_val = logreg_bow(pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_lemmatized")[:len(train)], 0.2)
accuracy, precision, recall, f1 = get_metrics(y_val, y_logistic_lemma)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))




def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer

def logreg_tfidf(data, valid_fraction):
    
    # Transform data to list
    list_corpus = data["text"].tolist()
    list_labels = data["target"].tolist()
    
    # Split train-validation data
    X_train, X_valid, y_train, y_valid = train_test_split(list_corpus, list_labels, 
                                                          test_size=valid_fraction, random_state=21)

    # Compute the tfidf vectors
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid)

    # Run the logistic regression model
    clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                             multi_class='multinomial', n_jobs=-1, random_state=21)
    clf_tfidf.fit(X_train_tfidf, y_train)

    y_predicted_tfidf = clf_tfidf.predict(X_valid_tfidf)
    
    # Cross validation score over 10 folds
    scores = cross_val_score(clf_tfidf, X_train_tfidf, y_train, cv=10)
    print("Cross validation over 10 folds: ", sum(scores)/10.)
    
    return y_predicted_tfidf, y_valid




print("Basic model")
y_predicted_tfidf, y_valid = logreg_tfidf(all_data[:len(train)], 0.2)

print("Lowercased model")
y_predicted_tfidf, y_valid = logreg_tfidf(pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_lowercased")[:len(train)], 0.2)

print("Normalized model")
y_predicted_tfidf, y_valid = logreg_tfidf(pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_normalized")[:len(train)], 0.2)

print("No stopwords model")
y_predicted_tfidf, y_valid = logreg_tfidf(pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_no_stopwords")[:len(train)], 0.2)

print("Lemmatized model")
y_predicted_tfidf, y_valid = logreg_tfidf(pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_lemmatized")[:len(train)], 0.2)




ts = time.time()

# Read the file with no stop words
clean_data = pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_no_stopwords")

# Lemmatize
clean_data['text'] = clean_data['text'].apply(lambda x: lemmatizer(nlp(x)))

# Lowercase
clean_data['text'] = clean_data['text'].apply(lambda x: x.lower())

clean_data.to_csv("tweets_clean", index=False)

print("Time spent: ", time.time() - ts)




# Define a tfidf and split function
def tfidf_split(data, valid_fraction):
    
    # Transform data to list
    list_corpus = data["text"].tolist()
    list_labels = data["target"].tolist()
    
    # Split train-validation data
    X_train, X_valid, y_train, y_valid = train_test_split(list_corpus, list_labels, 
                                                          test_size=valid_fraction, random_state=21)

    # Compute the tfidf vectors
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid)
    
    return X_train_tfidf, X_valid_tfidf, y_train, y_valid

X_train, X_valid, y_train, y_valid = tfidf_split(clean_data[:len(train)], 0.2)


# Define a general call for the different models
def get_cross_val(model, X_train, X_valid, y_train, y_valid):
    
    # Fit on train, predict on validation
    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    
    # Cross validation score over 10 folds
    scores = cross_val_score(clf, X_train, y_train, cv=10)
    print("Cross validation over 10 folds: ", sum(scores)/10.)
    
    return y_pred




ts = time.time()

print("Logistic regression: ")
y_LR = get_cross_val(LogisticRegression(C=30.0, class_weight='balanced', 
                                        solver='newton-cg', multi_class='multinomial', 
                                        n_jobs=-1, random_state=21), 
                                        X_train.toarray(), X_valid.toarray(), y_train, y_valid)

print("\nNaive Bayes:")
y_bayes = get_cross_val(GaussianNB(), X_train.toarray(), X_valid.toarray(), y_train, y_valid)

print("\nSVC:")
y_SVC = get_cross_val(LinearSVC(random_state=21, dual=False), X_train, X_valid, y_train, y_valid)

print("\nKNN:")
y_KNN = get_cross_val(KNeighborsClassifier(n_neighbors=15), X_train, X_valid, y_train, y_valid)

print("\nRandom forest:")
y_RF = get_cross_val(RandomForestClassifier(random_state=21), X_train, X_valid, y_train, y_valid)

print("\nXGBoost:")
y_XGB = get_cross_val(XGBClassifier(n_estimators=1000, random_state=21), X_train, X_valid, y_train, y_valid)

print("Time spent: ", time.time() - ts)




ts = time.time()
clean_data = pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_clean")
vectors = np.array([nlp(tweet.text).vector for idx, tweet in clean_data.iterrows()])
vectors.shape
print("Time spent: ", time.time() - ts)




# Center the vectors
vec_mean = vectors.mean(axis=0)
centered = pd.DataFrame([vec - vec_mean for vec in vectors])
print("Center shape: ", centered.shape)




ts = time.time()

def svc_model(vectors, train):
    # Split train-validation data
    X_train, X_valid, y_train, y_valid = train_test_split(vectors[:len(train)], train.target, 
                                                          test_size=0.2, random_state=21)

    # Create the LinearSVC model
    model = LinearSVC(random_state=21, dual=False)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Cross validation score over 10 folds
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print("Cross validation over 10 folds: ", scores, " --- ", sum(scores)/10.)
    
    # Uncomment to see model accuracy
    #print(f'Model test accuracy: {model.score(X_valid, y_valid)*100:.3f}%')
    
    return model

model_svc_basic = svc_model(centered, train)
print("Time spent: ", time.time() - ts)




# Submit results

y_test = model_svc_basic.predict(centered[-len(test):])
submission = pd.DataFrame({
    "id": test.id, 
    "target": y_test
})
submission.to_csv('submission_svc_basic.csv', index=False)




ts = time.time()
clean_data = all_data
vectors = np.array([nlp(tweet.text).vector for idx, tweet in clean_data.iterrows()])
vec_mean = vectors.mean(axis=0)
centered = pd.DataFrame([vec - vec_mean for vec in vectors])
model_svc_basic = svc_model(centered, train)
print("Time spent: ", time.time() - ts)




# Read the file with no stop words
clean_data = pd.read_csv("/kaggle/input/disaster-tweets-comp-introduction-to-nlp/tweets_clean")




# Example https://www.kaggle.com/reiinakano/basic-nlp-bag-of-words-tf-idf-word2vec-lstm

text_lengths = [len(x.split()) for x in (clean_data['text'])]
num_words = max(text_lengths)

def model_RNN(num_words, embed_dim, lstm_out, clean_data, batch_size, eta, dropout, n_epochs): 
    
    # Use the Keras tokenizer
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(clean_data['text'].values)

    # Pad the data 
    X = tokenizer.texts_to_sequences(clean_data['text'].values)
    X = pad_sequences(X, maxlen=num_words + 1)

    # Split data
    Y = pd.get_dummies(clean_data['target']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 21, stratify=Y)
    
    model = Sequential()
    model.add(Embedding(num_words, embed_dim, input_length = X.shape[1]))
    model.add(LSTM(lstm_out, recurrent_dropout=dropout, dropout=dropout))
    model.add(Dense(2,activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer=Adam(lr=eta), metrics = ['accuracy'])
    model_history = model.fit(X_train, Y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    
    return model, model_history

# Define RNN parameters
embed_dim = 64
lstm_out = 64 
batch_size = 16
eta = 0.001
dropout = 0.5
n_epochs = 20
n_reps = 10

#model_RNN_1, model_RNN_1_hist_rep = model_RNN(num_words, embed_dim, lstm_out, clean_data, batch_size, eta, dropout, n_epochs)




model_RNN_1_hist = [0]*n_epochs

for rep in range(n_reps):
    print("Repetition ", rep)
    model_RNN_1, model_RNN_1_hist_rep = model_RNN(num_words, embed_dim, lstm_out, clean_data, batch_size, eta, 0.2, n_epochs)
    model_RNN_1_hist = tuple(map(operator.add, model_RNN_1_hist, model_RNN_1_hist_rep.history['val_accuracy']))
    
model_RNN_1_hist = [x/n_reps for x in list(model_RNN_1_hist)] 




plt.plot(model_RNN_1_hist
plt.xlabel("Epochs")
plt.ylabel("Val_accuracy")
plt.xlim(0,20)




Xtest = tokenizer.texts_to_sequences(clean_data[-len(train):]['text'].values)
Xtest = pad_sequences(Xtest, maxlen=num_words + 1)

RNN = model.predict(Xtest)

y_test = model.predict(Xtest)
y_test = np.argmax(y_test,axis = 1)

submission = pd.DataFrame({
    "id": test.id, 
    "target": y_test
})
submission.to_csv('submission_lstm.csv', index=False)

