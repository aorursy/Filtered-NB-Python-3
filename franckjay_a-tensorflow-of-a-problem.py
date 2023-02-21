#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
'''
First, lets check out some interesting TF issues, because it is not 
simply Python. It has its own quirks. If you want to follow along 
a similar path to the one I am doing, go here: http://learningtensorflow.com/lesson2/
'''

x = tf.constant(5)
y = tf.Variable(x + 1)

print(y) #=> Tensor("Variable/read:0", shape=(), dtype=int32)
'''
Not exactly what one would be expecting, right? I thought it should be 6!
y is just an object here, not a number. If you change x to 5. instead of an 
integer, the dtype is transformed into float32 by the way. How do we get it to
run?
'''
#Turns everything into version that TF likes
model = tf.global_variables_initializer() 

#Now we start a session, which tells TF that we want to do something
with tf.Session() as session:
    session.run(model)
    print(session.run(y)) #This is what prints out 6

'''
Now that we finally have some output, let's do something more interesting by 
using an array or list:
'''

x=tf.constant([5,6,7])
with tf.Session() as session:
    session.run(model)
    print(session.run(y)) #Why did this print out 6 as well?!?
    
'''
We forgot to re-initialize the variables globally! As well as add y again as a function
'''  
print('Now with an array!')
x=tf.constant([5,6,7])
y = tf.Variable(x + 1)
model = tf.global_variables_initializer() 
with tf.Session() as session:
    session.run(model)
    print(session.run(y)) #Prints [6 7 8]
    
'''
However, we use larger variables. Let's try a bigger array and see 
the results
'''
dat=np.random.randint(100,size=10)
x=tf.constant(dat)
y = tf.Variable(x**2 +x+ 10)
model = tf.global_variables_initializer() 
with tf.Session() as session:
    session.run(model)
    print(session.run(y)) #Larger list of data.
    
    
'''
Thus far, we have been using x=constant, but we need not for eternity
'''    
print('Variable X')
x=tf.Variable(2.5)
y = tf.Variable(x**2 +x+ 10)
model = tf.global_variables_initializer() 
with tf.Session() as session:
    for i in range(10):
        x=x+1.0
        session.run(model)
        print(i,session.run(x)) #Larger list of data.




Now we are basically done trying out some TF basics. 
Let us create a NN to see if we can do it.




import pandas as pd
from sklearn.preprocessing import LabelBinarizer

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

'''
Let's separate out the labels and the features
'''
trainY=train[['label']]#First column
trainX=train.drop(['label'],axis=1)#Drop the labels
trainX=trainX.as_matrix()
test=test.as_matrix()
train=''#Save space

#Let's turn this into OneHot Encoding
#print (trainY.head(5))
enc = LabelBinarizer()
trainY=enc.fit_transform(trainY)
trainY=trainY.astype('float32')
#print (trainY[0:5][:])

print ('From:https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/?completed=/tensorflow-deep-neural-network-machine-learning-tutorial/ ')
print('Or: https://www.kaggle.com/jayantyadav/digit-recognizer/tensorflow-deep-nn')

nClass=10
nFeatures=784
batch_size = 128

graph = tf.Graph()
with graph.as_default():
  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset=tf.placeholder(tf.float32,shape=(batch_size,nFeatures))
  tf_train_labels=tf.placeholder(tf.float32, shape=(batch_size, nClass))
  tf_test_dataset = tf.constant(test)
  
  # Variables.
  weights = tf.Variable(tf.truncated_normal([nFeatures, nClass]))
  biases = tf.Variable(tf.zeros([nClass]))
  
  # Training computation.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)        
num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (trainY.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))





from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

print('Repurposed from: https://gist.github.com/vinhkhuc/e53a70f9e5c3f55852b0')
RANDOM_SEED=11
def init_weights(shape):
    """ Weight initialization """
    #Can also use Random normal if you prefer, but does slightly worse
    weights = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.relu(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_data():
    """ Read the data set and split them into training and validation sets """
    train = pd.read_csv("../input/train.csv")
    target=train['label']#First column
    trainX=train.drop(['label'],axis=1)#Drop the labels
    trainX=trainX.as_matrix()
    train=''#Save space

    # Prepend the column of 1s for bias
    N, M  = trainX.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = trainX
    
    #No Bias levels?
    #all_X = np.ones((N, M))
    #all_X[:, :] = trainX

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!

    return train_test_split(all_X, all_Y, test_size=0.05, random_state=RANDOM_SEED)
    #return trainX,test,target

def main():
    nEpochs=5 #Kaggle Times out after this number
    LR=1.0E-4 #Learning rate.
    h_size = 512 # Number of hidden nodes
    batch_size=1 #Process how many examples at once?
    Gradient=False#If False, uses AdamOptimizer
    regularize=True#Significantly improves the fit.
    
    print('Single Layer with N=',h_size,' nodes, a batch size of ',batch_size,'and learning rate=',LR)
    
    train_X, valid_X, train_y, valid_y = get_data()#Get the Validation, Train, Test data
    test=pd.read_csv("../input/test.csv")
    
    test=test.as_matrix() #We want this in matrix form, not Pandas dataframes
    N, M  = test.shape
    test_X = np.ones((N, M + 1))#Shape of the data and one Bias layer
    test_X[:, 1:] = test #Paste the test data on top
    #test_X = np.ones((N, M ))#Shape of the data and **NO** Bias layer
    #test_X[:, :] = test #Paste the test data on top
    test_X=test_X.astype('float32') #Make sure it is in the right data-type
    test=''#Save Space
    if regularize:
        train_X=train_X/np.max(train_X)
        test_X=test_X/np.max(train_X)
        valid_X=valid_X/np.max(train_X)
    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 784 features and 1 bias
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)
    #predict = tf.argmax( tf.matmul( tf.nn.relu(tf.matmul(X, w_1)), w_2), axis=1)
    

    
    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    if Gradient:
        updates = tf.train.GradientDescentOptimizer(LR).minimize(cost)
    else:
        updates = tf.train.AdamOptimizer(LR).minimize(cost)#Adam works much better.

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    print('Starting to Train')
    nBatches=int(float(len(train_X))/float(batch_size))
    for epoch in range(nEpochs):
        # Train with each example
        n=0
        #for i in range(len(train_X)):
        for i in range(nBatches):
            sess.run(updates, feed_dict={X: train_X[n:n+batch_size], y:train_y[n:n+batch_size]})
            n+=batch_size
        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        valid_accuracy  = np.mean(np.argmax(valid_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: valid_X, y: valid_y}))
        
        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * valid_accuracy))
        
    print('Done Training.')    
    train_X, valid_X, train_y, valid_y= '','','',''  #Save space for your father's sake  
    print ('Testing')
    predictions=sess.run(predict, feed_dict={X: test_X})
    sess.close()
    print('Done testing!')
    
    print('Prepare to submit!')
    image_id_n=np.arange(1,len(predictions)+1,1)
    subm=pd.DataFrame({'ImageId': image_id_n,'Label': predictions})
    print( subm.head(3))
    subm.to_csv('TF_out.csv',index=False)
    print('Done!')
main()    





import numpy as np
import tensorflow as tf

print('From: https://nathanbrixius.wordpress.com/2016/05/18/an-introduction-to-tensorflow/')

with tf.Session() as session:
    x = tf.placeholder(tf.float32, [1], name='x') # fed as input below
    y = tf.placeholder(tf.float32, [1], name='y') # fetched as output below
    b = tf.constant(1.0)
    y = x + b # here is our ‘model’: add one to the input.
    x_in = [2] # (2)
    y_final = session.run([y], {x: x_in}) # (3)
    print(y_final) # (4)

with tf.Session() as session:
    #Create placeholder variables
    X=tf.placeholder(tf.float32,[1])
    yhat=tf.placeholder(tf.float32,[1])
    #Create a bias variable
    b=tf.Variable(7.)
    m=tf.Variable(5.)
    #This is our equation/model
    yhat=m*X+b
    
    x_data=[2]
    tf.global_variables_initializer().run()

    print(session.run(yhat,{X:x_data }) )
    
    




import numpy as np
import tensorflow as tf

print('From: https://nathanbrixius.wordpress.com/2016/05/23/a-simple-predictive-model-in-tensorflow/')




import tensorflow as tf
def multilayer_perceptron(x):
    fc1 = layers.fully_connected(x, 256, activation_fn=tf.nn.relu)
    fc2 = layers.fully_connected(fc1, 256, activation_fn=tf.nn.relu)
    out = layers.fully_connected(fc2, 10, activation_fn=None)
    return out
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
pred = multilayer_perceptron(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
def train(session):
    batch_size = 200
    session.run(tf.initialize_all_variables())
    for epoch in range(10):
        epoch_loss = 0.0
        batch_steps = mnist.train.num_examples / batch_size
        for step in range(batch_steps):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = session.run([train_op, loss], {x: batch_x, y: batch_y})
            epoch_loss += c / batch_steps
        print ("Epoch %02d, Loss = %.6f" % (epoch, epoch_loss) )






print('From: https://github.com/nlintz/TensorFlow-Tutorials/blob/master/04_modern_net.py')

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED=1337
def get_data():
    add_Bias=True
    """ Read the data set and split them into training and validation sets """
    train = pd.read_csv("../input/train.csv")
    target=train['label']#First column
    trainX=train.drop(['label'],axis=1)#Drop the labels
    trainX=trainX.as_matrix()
    train=''#Save space

    
    #Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    
    if add_Bias:
        # Prepend the column of 1s for bias
        N, M  = trainX.shape
        all_X = np.ones((N, M + 1))
        all_X[:, 1:] = trainX
        return train_test_split(all_X, all_Y, test_size=0.05, random_state=RANDOM_SEED)
    else:   
        return train_test_split(trainX, all_Y, test_size=0.05, random_state=RANDOM_SEED)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): 
    # this network is the same as the previous one except with an extra hidden layer 
    # + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)


trX, valid_X, trY, valid_y = get_data()#Get the Validation, Train, Test data
#test=pd.read_csv("../input/test.csv")

regularize=True # This can help a lot.
if regularize:
    trX=trX/(np.max(trX)-np.min(trX) )
    #test_X=test_X/np.max(train_X)
    valid_X=valid_X/(np.max(valid_X)-np.min(valid_X) )

nExamples,nFeatures=np.shape(trX)
nLabels,nHidden=np.shape(trY)[1],625*1#Little change in performance

X = tf.placeholder("float", [None, nFeatures])
Y = tf.placeholder("float", [None, nLabels])


print (nExamples,nFeatures,nLabels)
w_h = init_weights([nFeatures, nHidden])
w_h2 = init_weights([nHidden, nHidden])
w_o = init_weights([nHidden, nLabels])

p_keep_input = tf.placeholder("float")#Percentage of X that is retained from dropout
p_keep_hidden = tf.placeholder("float")#Percentage of hidden layer that is kept 
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(10):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(valid_y, axis=1) ==
                         sess.run(predict_op, feed_dict={X: valid_X, 
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))






print('Added another layer to https://github.com/nlintz/TensorFlow-Tutorials/blob/master/04_modern_net.py')

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED=1
def get_data():
    add_Bias=True
    """ Read the data set and split them into training and validation sets """
    train = pd.read_csv("../input/train.csv")
    target=train['label']#First column
    trainX=train.drop(['label'],axis=1)#Drop the labels
    trainX=trainX.as_matrix()
    train=''#Save space

    
    #Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    
    if add_Bias:
        # Prepend the column of 1s for bias
        N, M  = trainX.shape
        all_X = np.ones((N, M + 1))
        all_X[:, 1:] = trainX
        return train_test_split(all_X, all_Y, test_size=0.05, random_state=RANDOM_SEED)
    else:   
        return train_test_split(trainX, all_Y, test_size=0.05, random_state=RANDOM_SEED)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_h2,w_h3, w_o, p_keep_input, p_keep_hidden): 
    # this network is the same as the previous one except with an extra hidden layer 
    # + dropout
    X = tf.nn.dropout(X, p_keep_input)#Drop some values
    h = tf.nn.relu(tf.matmul(X, w_h))#Multiply X by first layer, RELU

    h = tf.nn.dropout(h, p_keep_hidden)#Drop some more values
    h2 = tf.nn.relu(tf.matmul(h, w_h2))#Multiply by second layer, RELU

    h2 = tf.nn.dropout(h2, p_keep_hidden)#Drop even more
    h3 = tf.nn.relu(tf.matmul(h2,w_h3))
    
    h3 = tf.nn.dropout(h3,p_keep_hidden)
    
    return tf.matmul(h3, w_o)


trX, valid_X, trY, valid_y = get_data()#Get the Validation, Train, Test data
#test=pd.read_csv("../input/test.csv")

normalize=True # This can help a lot.
if normalize:
    trX=trX/(np.max(trX)-np.min(trX) )
    #test_X=test_X/np.max(train_X)
    valid_X=valid_X/(np.max(valid_X)-np.min(valid_X) )

nExamples,nFeatures=np.shape(trX)
nLabels,nHidden,nHidden2=np.shape(trY)[1],625*1,2000#Little change in performance

X = tf.placeholder("float", [None, nFeatures])
Y = tf.placeholder("float", [None, nLabels])


print (nExamples,nFeatures,nLabels)
#(NxM)(MxP)=(NxP)
w_h = init_weights([nFeatures, nHidden])
w_h2 = init_weights([nHidden,nHidden2])
w_h3 = init_weights([nHidden2, nHidden])
w_o = init_weights([nHidden, nLabels])

p_keep_input = tf.placeholder("float")#Percentage of X that is retained from dropout
p_keep_hidden = tf.placeholder("float")#Percentage of hidden layer that is kept 
py_x = model(X, w_h, w_h2,w_h3, w_o, p_keep_input, p_keep_hidden)

print('For Regularization: http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
regularize=True# Did worse on first step than others.
if regularize:
    beta=0.01
    #Mess with your cost function, for your father's sake
    regularizer = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_h2) +tf.nn.l2_loss(w_h3)+tf.nn.l2_loss(w_o)
    cost = tf.reduce_mean(cost + beta * regularizer)
    

train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(2):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(valid_y, axis=1) ==
                         sess.run(predict_op, feed_dict={X: valid_X, 
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))
print('Is not that helpful in the final analysis.')        

