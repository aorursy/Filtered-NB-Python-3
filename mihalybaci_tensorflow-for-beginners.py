#!/usr/bin/env python
# coding: utf-8



# Begin tutorial

import numpy as np
import tensorflow as tf

mnist = np.loadtxt("../input/train.csv",delimiter=",",skiprows=1) #first row are labels




# Turn true values for written numbers into one-hot arrays so that
# 0 = [1,0,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0,0]
# etc....

# For training set
nvalid = 10000  #set aside a certain number of images for validation tests
mnist_train = mnist[:mnist.shape[0]-nvalid,:]  
mnist_valid = mnist[mnist.shape[0]-nvalid:mnist.shape[0],:]

print(np.shape(mnist_train))
print(np.shape(mnist_valid))


num_val = np.array(mnist_train[:,0]).astype(np.int) # create array of true values, int for indexing
num_onehot = np.zeros((np.size(num_val),10)) #create array to store one-hot vectors
dummy_index = np.arange(0,np.size(num_val),1) #dummy index to help fill in num_onehot

num_onehot[dummy_index,num_val] = 1  #Use dummy index and true value to create one-hot

#Ensure that each number has proper one-hot representation
print(num_val[10:15])
print(num_onehot[10:15,:])




# Create array of flattened image data
image_data = mnist_train[:,1:]

#print(image_data[0,:])




# Set up model to perform analysis

x = tf.placeholder(tf.float32,[None,784]) # Assign a placeholder for our data of N dimension

W = tf.Variable(tf.zeros([784,10])) #Array of weights
b = tf.Variable(tf.zeros([10]))   #Array of biases

y = tf.nn.softmax(tf.matmul(x,W)+b) # Perform matrix multiplication to get probabilities




#Definte the cross-entropy function

y_ = tf.placeholder(tf.float32,[None,10]) # Array to contain the true values
#Basic version of cross-entropy
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#More numerically stable version of calculation
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))




train_step = tf.train.GradientDescentOptimizer(1.0e-4).minimize(cross_entropy)

sess = tf.InteractiveSession() # launch interactive session

tf.global_variables_initializer().run() #initialize variables

#Run the training step i times on j random images
steps = 2500
nimages = 1000

init = 0
final = nimages

# put images into batches and try not to take same image twice until all have been selected
for _ in range(steps):
    batch_xs = image_data[init:final,:] #take set of  images
    batch_ys = num_onehot[init:final,:] #take set true values
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys}) #run session
    init += nimages
    final += nimages
    if final >= np.size(num_val):
        batch_index = np.random.shuffle(np.arange(np.size(num_val))) #randomize indices
        init = 0
        final = nimages
   




# Organize data just as before

mnist_valid = mnist[mnist.shape[0]-nvalid:mnist.shape[0],:]
print(np.shape(mnist_valid))

num_valid = np.array(mnist_valid[:,0]).astype(np.int) # create array of true values, int for indexing
num_onehot_valid = np.zeros((np.size(num_valid),10)) #create array to store one-hot vectors
dummy_index_valid = np.arange(0,np.size(num_valid),1) #dummy index to help fill in num_onehot

num_onehot_valid[dummy_index_valid,num_valid] = 1  #Use dummy index and true value to create one-hot

# Create array of flattened image data
image_valid = mnist_valid[:,1:]


prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))  #Get True/False for predictions
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32)) #Get mean accuracy
print(sess.run(accuracy,feed_dict={x:image_valid, y_:num_onehot_valid}))




#For test set
mnist_test = np.loadtxt("../input/test.csv",delimiter=",",skiprows=1) #first row are labels

print(np.shape(mnist_test))

print('mnist_test({0[0]},{0[1]})'.format(mnist_test.shape))


# predict test set
#predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})


# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y,1)
keep_prob = tf.placeholder('float')

batches = 100
# using batches is more resource efficient
predicted_lables = np.zeros(mnist_test.shape[0])
for i in range(0,mnist_test.shape[0]//batches):
    predicted_lables[i*batches : (i+1)*batches] =         predict.eval(feed_dict={x: mnist_test[i*batches : (i+1)*batches], keep_prob: 1.0})


print('predicted_lables({0})'.format(len(predicted_lables)))

# output test image and prediction
#display(test_images[IMAGE_TO_DISPLAY])
#print ('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,predicted_lables[IMAGE_TO_DISPLAY]))

# save results
np.savetxt('submission_softmax.csv', 
           np.c_[range(1,len(mnist_test)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

