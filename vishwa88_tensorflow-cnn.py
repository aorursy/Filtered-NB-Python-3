#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import tensorflow as tf
test_dataset  = pd.read_csv('../input/test.csv',)
train = pd.read_csv('../input/train.csv', nrows = 40000)
valid = pd.read_csv('../input/train.csv',skiprows = 40000, nrows = 2000)

train_dataset = train.ix[:,1:785]
train_labels  = train.ix[:,0]

valid_dataset = valid.ix[:,1:785]
valid_labels  = valid.ix[:,0]




def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def shuffle(data, labels):
    rnd = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rnd)
    np.random.shuffle(labels)




train_dataset_np = (train_dataset.as_matrix()/255.0) - 1.0
train_labels_np = train_labels.as_matrix()

valid_dataset_np = (valid_dataset.as_matrix()/255.0) - 1.0
valid_labels_np = valid_labels.as_matrix()

test_np = (test_dataset.as_matrix()/255.0) - 1.0

shuffle(train_dataset_np, train_labels_np)




image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset_np, train_labels_np)
valid_dataset, valid_labels = reformat(valid_dataset_np, valid_labels_np)

test_dataset = test_np.reshape((-1,image_size, image_size, num_channels))    .astype(np.float32)




print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)




batch_size = 100
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))

  layer1_biases = tf.Variable(tf.zeros([depth]))

  layer2_weights = tf.Variable(tf.truncated_normal([patch_size,
                   patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size//4  * image_size//4  * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') # NO bias added here
    pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pool + layer1_biases)  
    
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
    pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(pool + layer2_biases)
    
    shape = hidden.get_shape().as_list()
    print(shape)
    reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] 
                                  * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))




num_steps = 2001
_step = 0
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    _step+=1
    if _step*batch_size > train_dataset.shape[0]: # Reshuffle data
       shuffle(train_dataset, train_labels)
       _step = 0
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :,:,:]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 300 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("test set evaluation")
  test_labels = np.argmax(test_prediction.eval(), axis=1)
  print( test_labels[0:3])




import matplotlib.pyplot as plt
k = 0 # Try different images indices k
plt.imshow(test_dataset[k,:,:,0])
plt.axis('off')
plt.show()
print("Label Prediction: %i"%test_labels[k])




submission = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})
submission.to_csv('submission.csv', index=False)
submission.head()

