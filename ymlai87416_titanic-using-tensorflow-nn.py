#!/usr/bin/env python
# coding: utf-8



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas

get_ipython().run_line_magic('matplotlib', 'inline')




#load training data
train_raw = pandas.read_csv('../input/train.csv', sep=',', index_col=0)

#load test data (with answer)
test_raw = pandas.read_csv('../input/test.csv', sep=',', index_col=0)

data_len = len(train_raw)
itest_len = data_len // 10
valid_len = data_len // 10 * 2
train_len = data_len - valid_len - itest_len

test_len = len(test_raw)

def from_panda_to_numpy(df, 
                        age_min, age_max, 
                        sibsp_min, sibsp_max, 
                        parch_min, parch_max,
                        fare_min, fare_max ):
    df_len = len(df)

    def convert_name_to_label(names_):
        lower_name = names_.str.lower()
        result = np.zeros(len(names_))
        result[lower_name.str.contains('mrs.')] = 1
        result[lower_name.str.contains('mr.')] = 2
        result[lower_name.str.contains('ms.')] = 3
        result[lower_name.str.contains('mlle.')] = 4
        result[lower_name.str.contains('miss.')] = 5
        result[lower_name.str.contains('sir.')] = 6
        result[lower_name.str.contains('rev.')] = 7
        result[lower_name.str.contains('mme.')] = 8
        result[lower_name.str.contains('master.')] = 9
        result[lower_name.str.contains('major.')] = 10
        result[lower_name.str.contains('lady.')] = 11
        result[lower_name.str.contains('jonkheer.')] = 12
        result[lower_name.str.contains('dr.')] = 13
        result[lower_name.str.contains('don.')] = 14
        result[lower_name.str.contains('col.')] = 15
        result[lower_name.str.contains('capt.')] = 16
        result[lower_name.str.contains('countess.')] = 17

        return result
    
    def convert_sex_to_label(sex_):
        lower_sex = sex_.str.lower()
        result = np.zeros(len(sex_))
        result[lower_sex == 'male'] = 0
        result[lower_sex == 'female'] = 1

        return result

    def convert_cabin_to_label(cabin_):
        lower_cabin = cabin_.str.lower()
        result = np.zeros(len(cabin_))
        lower_cabin = lower_cabin.replace(np.nan, '', regex=True)
        result[lower_cabin.str.contains('a')] = 1
        result[lower_cabin.str.contains('b')] = 2
        result[lower_cabin.str.contains('c')] = 3
        result[lower_cabin.str.contains('d')] = 4
        result[lower_cabin.str.contains('e')] = 5
        result[lower_cabin.str.contains('f')] = 6
        result[lower_cabin.str.contains('g')] = 7
        result[lower_cabin.str.contains('t')] = 8

        return result

    def convert_embark_to_label(embark_):
        lower_embark = embark_.str.lower()
        result = np.zeros(len(cabin_))
        result[lower_embark.str.contains('c')] = 1
        result[lower_embark.str.contains('q')] = 2
        result[lower_embark.str.contains('s')] = 3
        
    def clean_up_age(age_):
        result = np.zeros(len(age_), dtype=np.float32)
        result[:] = age_
        result[np.isnan(result)] = 0
        result[result < 1] *= 100 
        
        return result
    
    def clean_up_fare(fare_):
        result = np.zeros(len(fare_), dtype=np.float32)
        result[:] = fare_
        result[np.isnan(result)] = 0
        
        return result
        
        
    full_dataset = np.zeros((df_len, 32), dtype=np.float32)
    full_dataset[:, 0] = df["Pclass"] / 3
    salutaion_ = convert_name_to_label(df["Name"])
    full_dataset[:, 1] = salutaion_ == 1
    full_dataset[:, 2] = salutaion_ == 2
    full_dataset[:, 3] = salutaion_ == 3
    full_dataset[:, 4] = salutaion_ == 4
    full_dataset[:, 5] = salutaion_ == 5
    full_dataset[:, 6] = salutaion_ == 6
    full_dataset[:, 7] = salutaion_ == 7
    full_dataset[:, 8] = salutaion_ == 8
    full_dataset[:, 9] = salutaion_ == 9
    full_dataset[:, 10] = salutaion_ == 10
    full_dataset[:, 11] = salutaion_ == 11
    full_dataset[:, 12] = salutaion_ == 12
    full_dataset[:, 13] = salutaion_ == 13
    full_dataset[:, 14] = salutaion_ == 14
    full_dataset[:, 15] = salutaion_ == 15
    full_dataset[:, 16] = salutaion_ == 16
    full_dataset[:, 17] = salutaion_ == 17
    full_dataset[:, 18] = convert_sex_to_label(df["Sex"])
    full_dataset[:, 19] = (clean_up_age(df["Age"]) - age_min) / (age_max - age_min)
    full_dataset[:, 20] = (df["SibSp"] - sibsp_min) / (sibsp_max - sibsp_min)
    full_dataset[:, 21] = (df["Parch"] - parch_min) / (parch_max - parch_min)
    full_dataset[:, 22] = (clean_up_fare(df["Fare"]) - fare_min) / (fare_max - fare_min)
    carbin_type = convert_cabin_to_label(df["Cabin"])
    full_dataset[:, 23] = carbin_type == 0
    full_dataset[:, 24] = carbin_type == 1
    full_dataset[:, 25] = carbin_type == 2
    full_dataset[:, 26] = carbin_type == 3
    full_dataset[:, 27] = carbin_type == 4
    full_dataset[:, 28] = carbin_type == 5
    full_dataset[:, 29] = carbin_type == 6
    full_dataset[:, 30] = carbin_type == 7
    full_dataset[:, 31] = carbin_type == 8
    
    return full_dataset

age_min_ = min(0, train_raw["Age"].min())
age_max_ = train_raw["Age"].max()
sibsp_min_ = train_raw["SibSp"].min()
sibsp_max_ = train_raw["SibSp"].max()
parch_min_ = train_raw["Parch"].min()
parch_max_ = train_raw["Parch"].max()
fare_min_ = train_raw["Fare"].min()
fare_max_ = train_raw["Fare"].max()
    
full_dataset = from_panda_to_numpy(train_raw, 
                                   age_min_, age_max_, sibsp_min_, sibsp_max_, parch_min_, parch_max_, fare_min_, fare_max_)
full_label = np.zeros((data_len, 2), dtype=np.float32)
full_label[:, 0] = train_raw["Survived"]
full_label[:, 1] = 1-full_label[:, 0] 

test_dataset = from_panda_to_numpy(test_raw,
                                   age_min_, age_max_, sibsp_min_, sibsp_max_, parch_min_, parch_max_, fare_min_, fare_max_)




#utility function
def get_train_valid_set(full_dataset, full_label):
    rand_perm = np.random.permutation(len(full_dataset))
    dataset = full_dataset[rand_perm]
    label = full_label[rand_perm]
    return dataset[0:train_len], label[0:train_len] ,                 dataset[train_len:train_len+valid_len], label[train_len:train_len+valid_len],                 dataset[train_len+valid_len:], label[train_len+valid_len:]

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

train_dataset, train_labels, valid_dataset, valid_labels, itest_dataset, itest_labels     = get_train_valid_set(full_dataset, full_label)




hidden_layer_size_1=1024
hidden_layer_size_2=300
hidden_layer_size_3=50
input_size = 32
dropout = 0.7
num_labels = 2

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.constant(train_dataset)
    tf_train_labels = tf.constant(train_labels)
    tf_lambda = tf.placeholder(tf.float32)
    tf_dropout = tf.placeholder(tf.float32)
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_itest_dataset = tf.constant(itest_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    W1 = tf.Variable(
        tf.truncated_normal([input_size, hidden_layer_size_1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([hidden_layer_size_1]))
    W2 = tf.Variable(
        tf.truncated_normal([hidden_layer_size_1, hidden_layer_size_2], stddev=0.1))
    b2 = tf.Variable(tf.zeros([hidden_layer_size_2]))
    W3 = tf.Variable(
        tf.truncated_normal([hidden_layer_size_2, hidden_layer_size_3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([hidden_layer_size_3]))
    W4 = tf.Variable(
        tf.truncated_normal([hidden_layer_size_3, num_labels], stddev=0.1))
    b4 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    a1 = tf.nn.relu(tf.matmul(tf_train_dataset, W1) + b1)
    a2 = tf.nn.relu(tf.matmul(tf.nn.dropout(a1, tf_dropout), W2) + b2)
    a3 = tf.nn.relu(tf.matmul(tf.nn.dropout(a2, tf_dropout), W3) + b3)
    logits = tf.matmul(tf.nn.dropout(a3, tf_dropout), W4) + b4
    
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    loss += tf_lambda * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    
    valid_a1 = tf.nn.relu(tf.matmul(tf_valid_dataset, W1) + b1)
    valid_a2 = tf.nn.relu(tf.matmul(valid_a1, W2) + b2)
    valid_a3 = tf.nn.relu(tf.matmul(valid_a2, W3) + b3)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_a3, W4) + b4)
    
    test_a1 = tf.nn.relu(tf.matmul(tf_test_dataset, W1) + b1)
    test_a2 = tf.nn.relu(tf.matmul(test_a1, W2) + b2)
    test_a3 = tf.nn.relu(tf.matmul(test_a2, W3) + b3)
    test_prediction = tf.nn.softmax(tf.matmul(test_a3, W4) + b4)
    
    #internal test set
    itest_a1 = tf.nn.relu(tf.matmul(tf_itest_dataset, W1) + b1)
    itest_a2 = tf.nn.relu(tf.matmul(itest_a1, W2) + b2)
    itest_a3 = tf.nn.relu(tf.matmul(itest_a2, W3) + b3)
    itest_prediction = tf.nn.softmax(tf.matmul(itest_a3, W4) + b4)




assert not np.any(np.isnan(train_dataset))
assert not np.any(np.isnan(valid_dataset))
assert not np.any(np.isnan(itest_dataset))
assert not np.any(np.isnan(train_labels))
assert not np.any(np.isnan(valid_labels))
assert not np.any(np.isnan(itest_labels))

assert not np.any(np.isnan(test_dataset))


lambdas = [0.001]
dropsout = [0.5]

num_steps = 3001

error_trains = np.zeros((len(lambdas), len(dropsout)))
error_vals = np.zeros((len(lambdas), len(dropsout)))
error_test = np.zeros((len(lambdas), len(dropsout)))

for i, lambda_t in enumerate(lambdas) :
    for j, dropout in enumerate(dropsout):

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print("Initialized with lambda %f dropout %f" % (lambda_t, dropout))
            for step in range(num_steps):
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {tf_lambda : lambda_t, tf_dropout : dropout}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 500 == 0):
                    print("Train loss at step %d: %f" % (step, l))
                    print("Train accuracy: %.1f%%" % accuracy(predictions, train_labels))
                    print("Validation accuracy: %.1f%%" % accuracy(
                        valid_prediction.eval(), valid_labels))

            print("Train accuracy: %.1f%%" % accuracy(predictions, train_labels))
            print("Cross validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(itest_prediction.eval(), itest_labels))

            print("===== Model scoring =====")
            result = test_prediction.eval()

            error_trains[i, j] = 100-accuracy(train_prediction.eval(feed_dict), train_labels)
            error_vals[i, j] = 100-accuracy(valid_prediction.eval(feed_dict), valid_labels)
            error_test[i, j] = 100-accuracy(itest_prediction.eval(feed_dict), itest_labels)






