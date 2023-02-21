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




import re




import tensorflow as tf




from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 




get_ipython().system(' cat /kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv | head -n2')




get_ipython().system(' cat /kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv | wc -l')




get_ipython().system(' cat /kaggle/input/question-pairs-dataset/questions.csv | wc -l')




get_ipython().system(' cat /kaggle/input/question-pairs-dataset/questions.csv | head -n3')




tweets = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv",encoding="ISO-8859-1",header=None,names=["label","id","time","flag","user","text"]).sample(n=404351)
questions = pd.read_csv("/kaggle/input/question-pairs-dataset/questions.csv")




questions.head()




tweets.head()




# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
def preprocess_tweet(text):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    return text




tweets["text"] = tweets.text.apply(preprocess_tweet)




tweets.head()




tweets_train = tweets[["text","label"]].copy()
tweets_train["label"] = 0 # not a query
question_train = pd.DataFrame()
question_train["text"] = questions["question1"]
question_train["label"] = 1 # is a query

tweets_train_data = tweets_train.sample(5000)
question_train_data = question_train.sample(5000)

tweets_test_data = tweets_train[~tweets_train.index.isin(tweets_train_data.index)].sample(500)
question_test_data = tweets_train[~question_train.index.isin(question_train_data.index)].sample(500)

train = pd.concat([tweets_train_data,question_train_data],ignore_index=True)
train["text"] = train.text.apply(lambda x : x.replace("?",""))

test = pd.concat([tweets_test_data,question_test_data],ignore_index=True)
test["text"] = test.text.apply(lambda x : x.replace("?",""))




train[train.label == 1].sample()["text"].iloc[0]




train.shape




tokenizer = Tokenizer(oov_token="<OOV>")
corpus = train["text"].astype(str).tolist()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1




word_index = tokenizer.word_index




vocab_size = len(tokenizer.word_index)




embedding_dim = 100




total_words




tokenizer.texts_to_sequences(corpus[:10])




# pad sequences
input_sequences = tokenizer.texts_to_sequences(corpus)
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))




input_sequences[0]




train["label"]




# Note this is the 100 dimension version of GloVe from Stanford
# I unzipped and hosted it on my site to make this notebook easier
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt     -O /tmp/glove.6B.100d.txt')
embeddings_index = {};
with open('/tmp/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;




model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len,weights=[embeddings_matrix], trainable=False))
model.add(Bidirectional(LSTM(50, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())




# The patience parameter is the amount of epochs to check for improvement
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)




num_epochs = 50
history = model.fit(input_sequences, train["label"], epochs=num_epochs, validation_split=0.2, 
                    verbose=2,callbacks=[early_stop])

print("Training Complete")




import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])

plt.figure()




input_sequences = tokenizer.texts_to_sequences(test["text"])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))




model.evaluate(input_sequences,test["label"])






