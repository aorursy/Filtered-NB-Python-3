# libraries and variables

import math
import numpy as np
import pandas as pd

#To display plots in cell rather than in seperate window
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

# Reading the data
data = pd.read_csv("../input/train.csv")
print(data.shape)
print(data.head())

images = data.iloc[:,1:].values #Selecting subset, i.e, from column 1
images = images.astype(np.float) #Converting the values to float as in next step we divide by 255

#converting from (0,255) to (0,1)
images = np.multiply(images, 1.0/255.0)
print('images({0[0]},{0[1]})'.format(images.shape))

def display_image(row):
    image = row.reshape(28,28)
    plt.axis('off')
    plt.imshow(image, cmap=cm.binary) #To show the image in binary colors, black and white

display_image(images[6])

lables = data[[0]].values.ravel() #used to return a view, in contrast to returning a copy in flatten
print('Numbers of labels: {0}'.format(len(lables)))
print('label for [{0}] is: {1}'.format(8,lables[8]))

def flat_to_oneHot(lables_param,unique_label_Number):
    print(lables_param.shape[0])
    number_of_labels = lables_param.shape[0]
    #creating 10 spaces, for storing and marking the actual digit as 1
    index_offset = np.arange(number_of_labels) * unique_label_Number
    labels_one_hot = np.zeros((number_of_labels,unique_label_Number))
    labels_one_hot.flat[index_offset+lables_param.ravel()] = 1
    return labels_one_hot

labels_count = 10
lables_oneHot_format = flat_to_oneHot(lables,labels_count)

print('labels({0[0]},{0[1]})'.format(lables_oneHot_format.shape))
print('label[{0}] : {1}'.format(17,lables_oneHot_format[17]))

train_images = images
train_labels = lables_oneHot_format
print(train_labels)

#Initialization
tf.set_random_seed(4)
X = tf.placeholder(tf.float32, [None, 784])

#For the correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

#For learning rate
lr = tf.placeholder(tf.float32)

# test flag for batch normalization
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)

# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#here this NN consists of 3 layers
A = 32
B = 64
C = 1024

#Batch size
BATCH_SIZE = 100

#Initializing with small random values
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, A], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, tf.float32, [A]))
W2 = tf.Variable(tf.truncated_normal([5,5,A, B], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [B]))

W3 = tf.Variable(tf.truncated_normal([7*7*B, C], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [C]))
W4 = tf.Variable(tf.truncated_normal([C, 10], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

# The model

image1 = tf.reshape(X, [-1,28 , 28,1])
Y1C = tf.nn.conv2d(image1, W1, strides=[1, 1, 1, 1], padding='SAME') + B1
Y1bn, update_ema1 = batchnorm(Y1C, tst, iter, B1, convolutional=True)
Y1R = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1R, pkeep_conv, compatible_convolutional_noise_shape(Y1R))
P1 = max_pool(Y1)

Y2C = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2
Y2bn, update_ema2 = batchnorm(Y2C, tst, iter, B2, convolutional=True)
Y2R = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2R, pkeep_conv, compatible_convolutional_noise_shape(Y2R))
P2 = max_pool(Y2)


# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(P2, shape=[-1, 7 * 7 * B])
Y3C = tf.matmul(YY, W3) + B3
Y4bn, update_ema3 = batchnorm(Y3C, tst, iter, B3)
Y3R = tf.nn.relu(Y4bn)
Y3 = tf.nn.dropout(Y3R, pkeep)


Ylogits = tf.matmul(Y3, W4) + B4
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3)

#For calculating loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

#Calculating the accuaracy of trained model
train_predicted_output = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))#checks for equal indexes
accuracy = tf.reduce_mean(tf.cast(train_predicted_output, tf.float32))
#print(accuracy)

#Code for creating a new batch

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):

    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    #print(train_images[start:end])
    return train_images[start:end], train_labels[start:end]

    # training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# starting the tensor flow session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1500):

    #print(i)
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = next_batch(BATCH_SIZE)


    # learning rate decay
    max_learning_rate = 0.02
    min_learning_rate = 0.0001
    decay_speed = 1600
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1.0})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})

    #After the model is trained, lets test it on a test data set
test_images = pd.read_csv("../input/test.csv")
test_images = test_images.iloc[:,:].values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))
print(test_images.shape[0])

# predict test set
#predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})
predict = tf.argmax(Y,1)
# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0, test_images.shape[0] // BATCH_SIZE):
    predicted_lables[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = predict.eval(session=sess,
        feed_dict={X: test_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE],
                   pkeep: 1.0, tst: False, pkeep_conv: 1.0})

#print('predicted_lables({0})'.format(len(predicted_lables)))

display_image(test_images[275])
print('predicted_lables[{0}] => {1}'.format(11,predicted_lables[11]))

# saving the results
np.savetxt('Digit_Recogniser.csv',
           np.c_[range(1,len(test_images)+1),predicted_lables],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')

sess.close()

# This model achieved an accuracy of 0.99143
# Also note that, some part of the code has been adapted from Martin Gorner tutorial for deep learning and tensor flow