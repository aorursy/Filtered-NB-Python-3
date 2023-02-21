#!/usr/bin/env python
# coding: utf-8



import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import sys
import io


print(tf.test.gpu_device_name())
# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True




# Read in only the two columns we need 
emoji = pd.read_csv('../input/twitter_emoji.csv')

# remove rows with an emoji sequence of len 1
emoji = emoji[emoji.length > 1]

# check out the dataframe
emoji.head()




# get the relevent column & summary info
emojis = emoji.emoji_no_mods

n_messages = len(emojis) # number of lines
n_chars = len(' '.join(map(str, emojis))) 

print("Number of messages %d" % n_messages)
print("Their messages add up to %d characters" % n_chars)




# conjoin all emoji into one huge string
emojis = '\n'.join(map(str, emojis)).lower()

emojis[:100] # Show first 100 characters




# get the indices & counts of each character
chars = sorted(list(set(emojis)))
print('Count of unique characters (i.e., features):', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))




# set span to length of the shortest emoji sequence plus 1
# get all spans of that length, with a step width of 1
maxlen = emoji.emoji.str.len().min() + 1
step = 1
sentences = []
next_chars = []
for i in range(0, len(emojis) - maxlen, step):
    sentences.append(emojis[i: i + maxlen])
    next_chars.append(emojis[i + maxlen])
print('Number of sequences:', len(sentences), "\n")

print(sentences[:10], "\n")
print(next_chars[:10])




# splitting spans into character to predict & 
# preceding characters
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1




# define our model
model = Sequential()
model.add(LSTM(1, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# compile model
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)




def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # Function invoked for specified epochs. Prints generated text.
    # Using epoch+1 to be consistent with the training epochs printed by Keras
    if epoch+1 == 1 or epoch+1 == 15:
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(emojis) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = emojis[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            next_char = ""
            while next_char != "\n":
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
    else:
        print()
        print('----- Not generating text after Epoch: %d' % epoch)

generate_text = LambdaCallback(on_epoch_end=on_epoch_end)




get_ipython().run_line_magic('pinfo', 'ModelCheckpoint')




# define the checkpoint
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')

# fit model using our gpu
with tf.device('/gpu:0'):
    model.fit(x, y,
              batch_size=128,
              epochs=15,
              verbose=2,
              callbacks=[generate_text, checkpoint])




print(np.fromstring("💩😂"))




def generate_emoji_seq(input_chars, # seed characters for prediction
                       maxlen=maxlen, # max input length
                       chars=chars, # number of unique emoji
                       char_indices=char_indices # indices from one-hot encoding
                      ):
    generated = ''
    sentence = input_chars
    generated += sentence

    next_char = ''
    while next_char != '\n':
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, 1)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        
print (generate_emoji_seq("💩"))

