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




import json
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()




def get_data(fpath,numrows=-1):
    dd = collections.defaultdict(list)
    with open(fpath,'r') as fip:
        for i,line in enumerate(fip):
            jdata = json.loads(line)
            for key in jdata:
                dd[key].append(jdata[key])
            if numrows != -1 and i >= numrows:
                break
    df = pd.DataFrame(dd)
    return df




df_reviews = get_data(r'/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json',10000)
df_reviews.info()




df_reviews['stars'].hist()




x = df_reviews['stars'].sample(n=1000)
df_reviews['stars'].value_counts()




df_text_len = df_reviews['text'].sample(n=1000).apply(lambda x:len(x.split()))
df_text_len.hist()




from sklearn.model_selection import train_test_split
import tensorflow as tf




df_sample = df_reviews.sample(n=10000,random_state=42)
x = df_sample['text'].apply(lambda x:' '.join(x.split()[:300])).values
y = df_sample['stars']-1.0
y = y.values
print(len(x),len(y))




def tokenize(text):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000)
    lang_tokenizer.fit_on_texts(text)

    tensor = lang_tokenizer.texts_to_sequences(text)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')

    return tensor, lang_tokenizer

x_tensor,tokenizer = tokenize(x)




x_train,x_test,y_train,y_test = train_test_split(x_tensor,y,test_size=0.2,random_state=42)
print('#Train size: ',len(x_train),len(y_train))
print('#Test size: ',len(x_test),len(y_test))




BUFFER_SIZE_TRAIN = len(x_train)
BUFFER_SIZE_TEST = len(x_test)
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 128

steps_per_epoch_train = len(x_train)//BATCH_SIZE_TRAIN
steps_per_epoch_test = len(x_test)//BATCH_SIZE_TEST

embedding_dim = 256
units = 1024
vocab_inp_size = len(tokenizer.word_index)+1

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(BUFFER_SIZE_TRAIN)
train_dataset = train_dataset.batch(BATCH_SIZE_TRAIN, drop_remainder=True)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(BUFFER_SIZE_TEST)
test_dataset = test_dataset.batch(BATCH_SIZE_TEST,drop_remainder=True)




def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()




model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 512),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256,dropout=0.25,return_sequences=True)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256,dropout=0.25)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])




model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
model.summary()




history = model.fit(train_dataset,epochs=10,
                    validation_data=test_dataset)




## Avg MSE = 0.91
plot_graphs(history, 'loss')




from tqdm import tqdm
def pretrained_embeddings(file_path, EMBEDDING_DIM, VOCAB_SIZE, word2idx):
    # 1.load in pre-trained word vectors     #feature vector for each word
    print('Loading word vectors...')
    embed_dict = {}
    with open(os.path.join(file_path),  errors='ignore', encoding='utf8') as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            if word in word2idx:
                vec = np.asarray(values[1:], dtype='float32')
                embed_dict[word] = vec

    print('Found %s word vectors.' % len(embed_dict))

    # 2.prepare embedding matrix
    print('Filling pre-trained embeddings...')
    num_words = VOCAB_SIZE
    # initialization by zeros
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in tqdm(word2idx.items()):
        if i < VOCAB_SIZE:
            embedding_vector = embed_dict.get(word)
            if embedding_vector is not None:
              # words not found in embedding index will be all zeros.
              embedding_matrix[i] = embedding_vector

    return embedding_matrix


fpath = '/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'
embedding_matrix = pretrained_embeddings(fpath,200,len(tokenizer.word_index)+1,tokenizer.word_index)




model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1,200,embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),trainable=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128,dropout=0.25,return_sequences=True)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128,dropout=0.25,return_sequences=True)),
    tf.keras.layers.LayerNormalization(),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5)
])




model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()




history = model.fit(train_dataset,epochs=10,
                    validation_data=test_dataset)




## Severe Overfitting
plot_graphs(history, 'loss')




plot_graphs(history, 'accuracy')




class Encoder(tf.keras.Model):
   def __init__(self, vocab_size, embedding_dim, enc_units,embedding_matrix):
       super(Encoder,self).__init__()

       self.enc_units = enc_units
       
       self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
trainable=True)
       self.emb_dropout = tf.keras.layers.Dropout(0.25)
       
       self.gru_cell1 = tf.keras.layers.GRU(self.enc_units,
                              return_sequences=True,
                              return_state=True,
                              recurrent_initializer='glorot_uniform')
       self.gru1 = tf.keras.layers.Bidirectional(self.gru_cell1, merge_mode='concat',weights=None, backward_layer=None)
       self.layernorm1 = tf.keras.layers.LayerNormalization()
       
       self.gru_cell2 = tf.keras.layers.GRU(self.enc_units,
                              return_sequences=True,
                              return_state=True,
                              recurrent_initializer='glorot_uniform')
       self.gru2 = tf.keras.layers.Bidirectional(self.gru_cell2, merge_mode='concat', weights=None, backward_layer=None)
       self.layernorm2 = tf.keras.layers.LayerNormalization()
       
       
           
   def call(self, x):
       x = self.embedding(x)
       x = self.emb_dropout(x)
       
       output1, state_f, state_b = self.gru1(x)
       output1 = self.layernorm1(output1)
       output2, state_f, state_b = self.gru2(output1)
       output2 = self.layernorm2(output2)
       output = output1 + output2
       
       return output,output1,output2




class Attention(tf.keras.Model):
    def __init__(self,hidden_dim):
        super(Attention,self).__init__()
        self.hidden_dim = hidden_dim
        self.W1 = tf.keras.layers.Dense(hidden_dim)
        self.W2 = tf.keras.layers.Dense(hidden_dim)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self,q,k):
        # q is [hidden_dim] --> [1,1,hidden_dim]
        q = tf.expand_dims(q,0)
        q = tf.expand_dims(q,0)
        #print(q.shape)
        
        attn_score = self.V(tf.nn.tanh(self.W2(q) + self.W1(k)))
        attn_weights = tf.nn.softmax(attn_score,axis=1)
        return attn_weights




class Projection(tf.keras.Model):
    def __init__(self,hidden_dim,n_class):
        super(Projection,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        
        self.dense1 = tf.keras.layers.Dense(hidden_dim,activation='relu')
        self.fc_dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(n_class)
        
    def call(self,x):
        output = self.dense1(x)
        output = self.fc_dropout(output)
        output = self.dense2(output)
        return output




class Model(tf.keras.Model):
    def __init__(self,encoder,projection,attn,query_units):
        super(Model,self).__init__()
        self.encoder = encoder
        self.attn = attn
        self.project = projection
        self.query = self.add_weight(shape=(query_units,),
                             initializer='zeros',
                             trainable=True)
        
    def call(self,input):
        output,output1,output2 = self.encoder(input)
        #print(self.query.shape)
        attn_weights = self.attn(self.query,output)
        weighted_output = attn_weights * output
        weighted_sum = tf.reduce_sum(weighted_output,1)
        output = self.project(weighted_sum)
        return output




example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape




#Verify encoder
encoder = Encoder(len(tokenizer.word_index)+1,200,128,embedding_matrix)
sample_output,x,y = encoder(example_input_batch)
print(sample_output.shape,x.shape,y.shape)




#Verify Attention
attn = Attention(256)
attn_weights = attn(tf.ones(shape=(256,)),sample_output)
print(attn_weights.shape)




#Verify Attention weighted reduction
x = sample_output*attn_weights
print(x.shape)
x = tf.reduce_sum(x,1)
print(x.shape)

project = Projection(256,5)
x = project(x)
print(x.shape)




#Verify Entire Model output
model = Model(encoder,project,attn,256)
y = model(example_input_batch)
print(x.shape,y.shape)




#__init__(self, vocab_size, embedding_dim, enc_units,embedding_matrix):
encoder = Encoder(len(tokenizer.word_index)+1,200,128,embedding_matrix)
attn = Attention(256)
projection = Projection(256,5)
model = Model(encoder,projection,attn,256)

sample_output = model(example_input_batch)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))




@tf.function
def train_step(inp, targ,model,loss_function,optimizer):
    loss = 0
    with tf.GradientTape() as tape:
        y_pred = model(inp)
        batch_loss = loss_function(targ,y_pred)

    variables = model.trainable_variables
    gradients = tape.gradient(batch_loss,variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss,tf.argmax(y_pred,axis=1)

@tf.function
def val_step(inp, targ,model,loss_function):
    y_pred = model(inp)
    batch_loss = loss_function(targ,y_pred)
    return batch_loss,tf.argmax(y_pred,axis=1)




loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()




import time
EPOCHS = 10
training_record = pd.DataFrame(columns = ['epoch', 'training_loss', 'validation_loss', 'epoch_time'])

train_accuracy = tf.keras.metrics.Accuracy()
val_accuracy = tf.keras.metrics.Accuracy()

template_loss = 'batch {} ============== train_loss: {}'
template_acc = 'batch {} ============== train_accuracy: {}'
template = 'Epoch {}/{}'

for epoch in range(EPOCHS):
    print(template.format(epoch +1,EPOCHS))
    
    train_accuracy.reset_states()
    val_accuracy.reset_states()
    
    start = time.time()

    total_val_loss = 0
    total_train_loss = 0
    
    for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch_train)):
        batch_loss,y_pred = train_step(inp, targ, model,loss_function,optimizer)
        total_train_loss += batch_loss
        train_accuracy.update_state(targ,y_pred)
        
        if batch % 10 == 0:
            print(template_loss.format(batch +1,
                            round(batch_loss.numpy(),4)))
            print(template_acc.format(batch +1,
                            round(train_accuracy.result().numpy(),4)))
            
    total_train_loss = total_train_loss/steps_per_epoch_train
    print("Total train loss: ",total_train_loss.numpy())
    print("Total train accuracy ",train_accuracy.result().numpy())
    
    total_val_loss = 0
    for (batch, (inp, targ)) in enumerate(test_dataset.take(steps_per_epoch_test)):
        batch_loss,y_pred = val_step(inp, targ, model,loss_function)
        total_val_loss += batch_loss
        val_accuracy.update_state(targ,y_pred)
        
        if batch % 10 == 0:
            print(template_loss.format(batch +1,
                            round(batch_loss.numpy(),4)))
            print(template_acc.format(batch +1,
                            round(val_accuracy.result().numpy(),4)))
    
    total_val_loss = total_val_loss/steps_per_epoch_test
    
    print("Total Validation loss: ",total_val_loss.numpy())
    print("Total Validation accuracy ",val_accuracy.result().numpy())

