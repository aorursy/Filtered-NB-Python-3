#!/usr/bin/env python
# coding: utf-8



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




try:
  # %tensorflow_version only exists in Colab.
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass




import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))




import numpy as np
import tensorflow as tf




import pandas as pd
headlines = pd.read_csv('/kaggle/input/million-headlines/abcnews-date-text.csv')['headline_text']




headlines[0]




import itertools as it 

def sliding_window(txt):
    for i in range(len(txt) - 2):
        yield txt[i], txt[i + 1], txt[i + 2]

window = list(it.chain(*[sliding_window(_) for _ in headlines[:10000]]))




for win in window[:10]:
    print(win)




mapping = {c: i for i, c in enumerate(pd.DataFrame(window)[0].unique())}
integers_in = np.array([[mapping[w[0]], mapping[w[1]]] for w in window])
integers_out = np.array([mapping[w[2]] for w in window]).reshape(-1, 1)




mapping




integers_in.shape




from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.models import Sequential

num_letters = len(mapping) # typically 36 -> 26 letters + 10 numbers

# this one is so we might grab the embeddings
model_emb = Sequential()
embedding = Embedding(num_letters, 2, input_length=2)
model_emb.add(embedding)
output_array = model_emb.predict(integers_in)
output_array.shape




import matplotlib.pylab as plt

idx_to_calc = list(mapping.values())
idx_to_calc = np.array([idx_to_calc, idx_to_calc]).T

translator = {v:k for k,v in mapping.items()}
preds = model_emb.predict(idx_to_calc)

plt.scatter(preds[:, 0, 0], preds[:, 0, 1], alpha=0)
for i, idx in enumerate(idx_to_calc):
    plt.text(preds[i, 0, 0], preds[i, 0, 1], translator[idx[0]])




from tensorflow.keras.optimizers import Adam

# this one is so we might learn the mapping
model_pred = Sequential()
model_pred.add(embedding)
model_pred.add(Flatten())
model_pred.add(Dense(num_letters, activation="softmax"))

adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model_pred.compile(adam, 'categorical_crossentropy', metrics=['accuracy'])

output_array = model_pred.predict(integers_in)
output_array.shape




from sklearn.preprocessing import OneHotEncoder

to_predict = OneHotEncoder(sparse=False).fit_transform(integers_out)
model_pred.fit(integers_in, to_predict, epochs=10, verbose=1)




preds = model_emb.predict(idx_to_calc)
plt.scatter(preds[:, 0, 0], preds[:, 0, 1], alpha=0)
for i, idx in enumerate(idx_to_calc):
    plt.text(preds[i, 0, 0], preds[i, 0, 1], translator[idx[0]])






