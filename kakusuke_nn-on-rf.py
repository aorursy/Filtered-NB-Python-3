
import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

import numpy as np

n_sample = 38000

train = train.reindex(np.random.permutation(train.index))
train = train.reset_index(drop=True)

train_images = train[:n_sample].drop('label', 1)
train_labels = train[:n_sample]['label']
cv_images = train[n_sample:].drop('label', 1)
cv_labels = train[n_sample:]['label']


from sklearn.ensemble import RandomForestClassifier

n_estimators = 10#64

clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=None, max_features=0.05, max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=n_estimators,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
clf.fit(train_images, train_labels)

estimated = pd.DataFrame([e.predict(test)for e in clf.estimators_]).T


estimated

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 794])
W = tf.Variable(tf.zeros([794, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.000).minimize(cross_entropy)

init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(init)
actuals = []

n_epoc = 5
n_batch = 100
loop = 0

def to_feed(images, labels=None, rf=0.8):
    predicted = pd.get_dummies(clf.predict(images))
    x = pd.concat([(images.reset_index(drop = True) - 126) * 1.0 / 255.0, predicted * rf], 1)
    if labels is not None:
      y_ = pd.get_dummies(labels)
      return(x, y_)
    return (x,)


for epoc in range(n_epoc):
  xs, ys_ = to_feed(train_images, train_labels)

  permutation = np.random.permutation(xs.index)
  xs.reindex(permutation)
  ys_.reindex(permutation)
  xs.reset_index(drop=True)
  ys_.reset_index(drop=True)
  for i in range(int(n_sample / n_batch)):
    index = (xs.index >= i * n_batch) & (xs.index < (i + 1) * n_batch)
    sess.run(train_step, feed_dict={x: xs.loc[index], y_: ys_.loc[index]})
  
  actuals.append([loop * n_epoc + epoc, 'training',  sess.run(accuracy, feed_dict={x: xs, y_: ys_})])
  cv_xs, cv_ys_ = to_feed(cv_images, cv_labels)
  actuals.append([loop * n_epoc + epoc, 'validation', sess.run(accuracy, feed_dict={x: cv_xs, y_: cv_ys_})])
  cv_xs, cv_ys_ = to_feed(cv_images, cv_labels, 1.0)
  actuals.append([loop * n_epoc + epoc, 'validation with full rf output', sess.run(accuracy, feed_dict={x: cv_xs, y_: cv_ys_})])
  cv_xs, cv_ys_ = to_feed(cv_images, cv_labels, 0)
  actuals.append([loop * n_epoc + epoc, 'validation without rf output', sess.run(accuracy, feed_dict={x: cv_xs, y_: cv_ys_})])
    
loop = loop + 1

cv_xs, cv_ys_ = to_feed(0, cv_images, cv_labels)
sess.run(accuracy, feed_dict={x: cv_xs, y_: cv_ys_})
 

import seaborn as sns
sns.lmplot(x='epoc', y='accuracy', hue='type', data=pd.DataFrame(actuals, columns=['epoc', 'type', 'accuracy']))

predict = tf.argmax(y,1)
predication = pd.DataFrame(sess.run(predict, feed_dict=to_feed(test), columns=['Label'], index=(test.index + 1))
predication.to_csv('output.csv', index=True, index_label='ImageId')


