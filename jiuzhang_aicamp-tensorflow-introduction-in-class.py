#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score




x = # TODO
y = # TODO
f = # TODO




sess = # TODO
sess.run(x.initializer)
sess.run(y.initializer)
result = #TODO
print(result)




sess.close()




# TODO




print(result)




#TODO




print(result)




init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(x.eval())
    print(y.eval())
    result = # TODO




print(result)





def plot_decision_boundary(X, model):
    h = .02 

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))


    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)


def f(X):
    """
    input: x
    output: y = 3x + 4
    """
    return 3*X + 4





N = # TODO
noise_level = #TODO
trainX = np.linspace(-4.0, 4.0, N)
np.random.shuffle(trainX)
trainY = # TODO


learning_rate = # TODO
training_epochs = # TODO
display_step = # TODO




plt.scatter(trainX, trainY)




from sklearn.base import BaseEstimator




class LinearRegressionTF(BaseEstimator):
    def __init__(self, learning_rate, training_epochs, display_step, annotate=False):
        self.annotate = annotate
        self.sess = tf.Session()
        self.training_epochs = # TODO
        self.learning_rate = # TODO
        self.display_step = # TODO
        
        
    def fit(self, trainX,trainY):
        N = # TODO  总共多少个数据
        # 图的输入
        self.X = # TODO
        self.Y = # TODO
        
        
        # 参数的定义
        self.W = # TODO
        self.b = # TODO
        
        # 线性模型
        self.pred =# TODO
        
        # mean squre error
        cost = # TODO
        
        # 优化器
        optimizer = # TODO
        
        # 初始化所有的参数
        init = # TODO
        self.sess.run(init)

        
        if self.annotate:
            plt.plot(trainX, trainY, 'ro', label='Original data')
            plt.plot(trainX, # TODO), label='Fitted line')
            plt.legend()
            plt.title("This is where model starts to learn!!")
            plt.show()
            
        # 训练开始
        for epoch in range(self.training_epochs):
            for (x, y) in zip(trainX, trainY):
                # TODO

            #展示训练结果
            if (epoch+1) % display_step == 0:
                c = # TODO
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
            "W=", self.sess.run(self.W), "b=", self.sess.run(self.b))
                
            #显示拟合的直线
                if self.annotate:
                    plt.plot(trainX, trainY, 'ro', label='Original data')
                    plt.plot(trainX, self.sess.run(self.W) * trainX + self.sess.run(self.b), label='Fitted line')
                    plt.legend()
                    plt.show()
                #plt.pause(0.5)

        print("Optimization Finished!")
        training_cost = # TODO
        print("Training cost=", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n')

        
    def predict(self, testX):
        prediction = # TODO
        return prediction
    
    def score(self, testX, testY):
        result = # TODO
        return r2_score(testY, result)
        
    




lr = # TODO
lr.fit(trainX, trainY)




from sklearn.model_selection import cross_val_score
cross_val_score(lr, trainX, trainY, cv=5).mean()









tf.reset_default_graph()




N = 100
D = 2
trainX = np.random.randn(N, D)

delta = 1.75
trainX[:N//2] += np.array([delta, delta])
trainX[N//2:] += np.array([-delta, -delta])

trainY = np.array([0] * (N//2) + [1] * (N//2))
plt.scatter(trainX[:,0], trainX[:,1], s=100, c=trainY, alpha=0.5)
plt.show()




original_label = np.array([0] * (N//2) + [1] * (N//2))




from sklearn.metrics import accuracy_score
class LogisticRegressionTF(BaseEstimator):
    def __init__(self, learning_rate, training_epochs, display_step, annotate=False):
        self.annotate = annotate
        self.sess = tf.Session()
        self.training_epochs = # TODO
        self.learning_rate = # TODO
        self.display_step = # TODO
        
        
    def fit(self, trainX,trainY):
        N, D = # TODO
        _, c = # TODO
        # 图的输入
        self.X = # TODO
        self.Y = # TODO
        
        
        # 参数的定义
        self.W = # TODO
        self.b = # TODO
        
        # logistic prediction
        #self.pred = tf.sigmoid(tf.add(tf.matmul(self.X, self.W), self.b))
        output_logits = #TODO
        self.pred = #TODO   # turn logits to probability
        
        # 交叉熵loss
        #cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
        cost= # TODO
        
        # 优化器
        optimizer = # TODO
        
        # 初始化所有的参数
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        
        # 可视化初始化的模型边界
        if self.annotate:
            assert len(trainX.shape) == 2, "Only 2d points are allowed!!"

            plt.scatter(trainX[:,0], trainX[:,1], s=100, c=original_label, alpha=0.5) 

            h = .02 
            x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
            y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
            plt.title("This is where model starts to learn!!")
            plt.show()

        

        # 训练开始
        for epoch in range(self.training_epochs):
            for (x, y) in zip(trainX, trainY):
                # TODO

            #展示训练结果
            if (epoch+1) % display_step == 0:
                c = # TODO
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),             "W=", self.sess.run(self.W), "b=", self.sess.run(self.b))
                
            #显示拟合的直线
                if self.annotate:
                    assert len(trainX.shape) == 2, "Only 2d points are allowed!!"

                    plt.scatter(trainX[:,0], trainX[:,1], s=100, c=original_label, alpha=0.5) 
             
                    h = .02 
                    x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
                    y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

                    Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
                    plt.show()



        print("Optimization Finished!")
        training_cost = # TODO
        print("Training cost=", training_cost, "W=", self.sess.run(self.W), "b=", self.sess.run(self.b), '\n')

        
    def predict(self, testX):
        prediction = # TODO
        return #TODO
    
    def score(self, testX, testY):
        # suppose the testY has been one hot encoded
        #eg:#0: [1,0]  -> 0, 0
            #1: [1,0]  -> 1, 0
            #2: [0,1]  -> 2, 1
        _ , true_result = # TODO
        result = # TODO
        return accuracy_score(true_result, result)




from sklearn.preprocessing import OneHotEncoder




le = #TODO
le.fit(trainY.reshape(N,-1))
trainY = # TODO




logisticTF = #TODO
logisticTF.fit(trainX, trainY)




from sklearn.model_selection import cross_val_score
cross_val_score(logisticTF, trainX, trainY, cv=5).mean()




tf.reset_default_graph()




#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#trainX, trainY = mnist.train.next_batch(5000) #5000个数据作为近邻集合
#testX, testY = mnist.test.next_batch(200) #200个数据用于测试









#data_folder = "../input/ninechapter-digitsub"
data_folder = "data"
trainX = np.genfromtxt(os.path.join(data_folder, "digit_mnist_trainx.csv"), delimiter=',')
trainY = np.genfromtxt(os.path.join(data_folder, "digit_mnist_trainy.csv"), delimiter=',')
testX = np.genfromtxt(os.path.join(data_folder, "digit_mnist_testx.csv"), delimiter=',')
testY = np.genfromtxt(os.path.join(data_folder, "digit_mnist_testy.csv"), delimiter=',')




xtr = # TODO
xte = # TODO




distance = # TODO

# train 0: [1,...1]
# train 1: [0,...0]
# test : [1,...1]
# tf.subtract(xtr, xte):
#  0: [0,...0]
#  1: [-1,...-1]
# tf.square:
#  0: [0,...0]
#  1: [1,...1]
#  tf.reduce_sum(tf.square(tf.subtract(xtr, xte)), reduction_indices=1):
#  0: [0]
#  1: [784]




# 因为是topk大的值，这里distance取负号
KVALUE = # TODO
pred= #TODO




from collections import Counter
accuracy = 0.

# 初始化参数
init = # TODO


# 开始训练
with tf.Session() as sess:
    sess.run(init)

    # 预测测试数据的标签 (passive learner)
    for i in range(len(testX)):
        # 最近邻的序号
        values, knn_index = # TODO

        # 拿到k个邻居后做全民公投，得票最多的为预测标签
        c = #TODO
        result = # TODO
        # 计算最近邻的标签和真实标签值
        print("Test", i, "Prediction:", result,             "True Class:", np.argmax(testY[i]))
        # 正确率
        if result == np.argmax(testY[i]):

            accuracy = # TODO
    print("Done!")
    print("Accuracy:", accuracy)




tf.reset_default_graph()




N = # TODO
D = 2
trainX = np.random.randn(N, D)

delta = 2
#trainX[:N//3] += np.array([delta, delta])
#trainX[N//3:N*2//3] += np.array([-delta, delta])
#trainX[N*2//3:] += np.array([0, -delta])


delta = 1.75
trainX[:N//2] += np.array([delta, delta])
trainX[N//2:] += np.array([-delta, -delta])

trainY = np.array([0] * (N//2) + [1] * (N//2))
plt.scatter(trainX[:,0], trainX[:,1], s=100, c=trainY, alpha=0.5)
plt.show()




from sklearn.metrics import accuracy_score




from matplotlib import colors
from sklearn.utils.fixes import logsumexp


class NaiveBayesTF(BaseEstimator):
    
    def __init__(self):
        self.dist = #TODO
        self.sess = #TODO

    def fit(self, trainX, trainY):
        # Separate training points by class (nb_classes * nb_samples * nb_features)
        unique_classes = # TODO
        points_by_class = #TODO
        
        input_x = # TODO
        # 估计每个类底下每一种feature的均值和方差
        # shape: num_classes * nb_features
        
        moments =# TODO
        mean, var = # TODO
        #print(mean.shape)
        #print(var.shape)
        
        # 点集实验里为2类，每个数据点有2个特征 
        # known mean and variance
        self.dist = #TODO
        

    def predict(self, testX):
        assert self.dist is not None
        num_classes, num_features = map(int, self.dist.scale.shape)

        # 条件概率 log P(x|c)
        # (nb_samples, nb_classes)
        cond_probs = # TODO
        
        # 第一个点: 2.0,3.5
        # 第二个点: 0.5,1.4
        # tf.tile (num_classes = 2):
        # 第一个点: 2.0,3.5,2.0,3.5
        # 第二个点: 0.5,1.4,0.5,1.4
        # tf.reshape:
        # 第一个点: 2.0,3.5 
        #         2.0,3.5
        # 第二个点：0.5,1.4
        #         0.5,1.4

        # P(C) 均匀分布
        priors =#TODO

        # 后验概率取log, log P(C) + log P(x|C)
        posterior = # TODO
        
        # 取概率最大的那一个
        result = # TODO

        return result
    
    
    def score(self, testX, testY):
        result = self.predict(testX)
        return accuracy_score(testY, result)




tf_nb = # TODO
tf_nb.fit(trainX, trainY)




x_min, x_max = trainX[:, 0].min() - .5, trainX[:, 0].max() + .5
y_min, y_max = trainX[:, 1].min() - .5, trainX[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                     np.linspace(y_min, y_max, 30))




Z = tf_nb.predict(np.c_[xx.ravel(), yy.ravel()])




Z = Z.reshape(xx.shape)




plt.scatter(trainX[:,0], trainX[:,1], s=100, c=trainY, alpha=0.5)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()




#tf_nb.score(trainX, trainY)














from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
import pandas as pd
import numpy as np
import os




#data_folder = "../input"
data_folder = "./data"
train_data = pd.read_csv(os.path.join(data_folder, "fashion-mnist_train.csv.zip"))
test_data = pd.read_csv(os.path.join(data_folder, "fashion-mnist_test.csv.zip"))




trainX = np.array(train_data.iloc[:, 1:])
trainY = np.array(train_data.iloc[:, 0])
testX = np.array(test_data.iloc[:, 1:])
testY = np.array(test_data.iloc[:, 0])




IMAGE_CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}




import matplotlib.pyplot as plt
img_size = 28
for img, label in zip(trainX[:10], trainY[:10]):
    # TODO
    #plt.title(IMAGE_CLASSES[label])
    #plt.show()




# 参数设定
#The 10 categories
#784 Each image is 28x28 pixels
num_steps = #TODO# Total steps to train
batch_size = # TODO # The number of samples per batch
num_trees = # TODO
max_nodes = # TODO









tf.reset_default_graph()
class RandomForestTF(BaseEstimator):
    
    def __init__(self, num_trees):
        self.num_trees = # TODO
        
    def fit(self, X, Y, num_steps, batch_size,max_nodes):
        num_classes = # TODO   #len(IMAGE_CLASSES)
        num_data = # TODO
        num_features = # TODO
        
        self.X = # TODO
        self.Y = # TODO 
        
        
        # 随机森林的参数
        hparams = # TODO
        
        
        # 随机森林的计算图
        forest_graph = # TODO
        
        train_operation = # TODO
        loss_operation = # TODO
        
        # inference_graph will return probabilities, decision path and variance
        self.infer_op, _, _ = # TODO
        correct_prediction = # TODO
        self.accuracy_op = # TODO
        
        
        # 将初始化的操作和树的参数初始化 作为一个整体操作
        init_vars = # TODO

        
        self.sess = tf.Session()
        self.sess.run(init_vars)

        # 开始训练
        cnt = 0
        for i in range(1, num_steps + 1):
            # Prepare Data
            # 每次学习一个batch的MNIST data
            #batch_x, batch_y = training_set.next_batch(batch_size)
            start, end = ((i-1) * batch_size) % num_data, (i * batch_size) % num_data
            
            batch_x, batch_y = # TODO
            _, l = # TODO
            if i % 50 == 0 or i == 1:
                acc = # TODO
                print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))
                
    def predict(self, testX):
        results = # TODO
        return # TODO
    
    def score(self, testX, testY):
        accuracy = # TODO
        return accuracy




rftf = # TODO




#rftf.fit(trainX, trainY, num_steps, batch_size, max_nodes)




#rftf.score(testX, testY)






