#!/usr/bin/env python
# coding: utf-8



import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
import os
import time




class BaseModel:
    def __init__(self, config):
        self.config = config
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()
        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='placeholder')

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        self.saver.save(sess, self.config['train_config']['checkpoint_dir'] + '/checkpoint.ckpt', self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config['train_config']['checkpoint_dir'])
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(1, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)
            

    def init_saver(self):
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError




class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.set_model_attr()
        self.model.build_model()
        
#         self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.init = tf.global_variables_initializer()
        
        self.sess.run(self.init)# init variables
        # init iterator for dataAPI
        self.sess.run(self.data.iterator.initializer, feed_dict={self.data.data_placeholder: self.data.train}) 
        
    def set_model_attr(self):
        raise NotImplementedError


    def train(self):
        num_epochs = self.config['train_config']['num_epochs']
        learning_rate = self.config['model_config']['learning_rate']
        lastest_step = sess.run(self.model.global_step_tensor)
        for cur_epoch in tqdm(range(lastest_step, lastest_step + num_epochs)):
            lr = learning_rate if cur_epoch > num_epochs / 2 else learning_rate * 3
            self.train_epoch(cur_epoch, lr)
            self.sess.run(self.model.increment_cur_epoch_tensor)
        self.model.save(self.sess)
        self.logger.view_board()

    def train_epoch(self, epoch_index, lr):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError




class DataProvider:
    def __init__(self, config):
        self.config = config
        
        if 'dataset_name' in self.config['data_config']:
            self.train, self.test = self.load_data()
        else:
            df = self.read_data()
            self.train, self.test = train_test_split(df, test_size=0.2)
        self.n_examples, self.n_features = self.train.shape

        self.data_placeholder = tf.placeholder(tf.float32, self.train.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices(self.data_placeholder)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.config['data_config']['minibatch_size']).repeat(self.config['train_config']['num_epochs'])
        self.iterator = train_dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()
    def read_data(self):
        data = pd.read_csv(self.config['data_config']['file_path'], sep=self.config['data_config']['seperator'], 
                           names=['user_id','item_id','rate','time_stamp'], engine='python', header=0)

        user_c = CategoricalDtype(sorted(data['user_id'].unique()), ordered=True)
        item_c = CategoricalDtype(sorted(data['item_id'].unique()), ordered=True)
        row = data['item_id'].astype(item_c).cat.codes
        col = data['user_id'].astype(user_c).cat.codes

        sparse_matrix = csr_matrix((data['rate'], (row,col)), shape=(item_c.categories.size, user_c.categories.size))
        df = pd.SparseDataFrame(sparse_matrix, columns=user_c.categories, index=item_c.categories, default_fill_value=0)        
        return df
    
    def load_data(self):
        train = load_npz('../input/movielens-dataset-split/{}-train.npz'.format(self.config['data_config']['dataset_name']))
        test = load_npz('../input/movielens-dataset-split/{}-test.npz'.format(self.config['data_config']['dataset_name']))
        train_df = pd.SparseDataFrame(train, default_fill_value=0)
        test_df = pd.SparseDataFrame(test, default_fill_value=0)
        return train_df, test_df




class DeepAutoEncoder(BaseModel):
    def __init__(self, config, pretrained_models=None):
        super(DeepAutoEncoder, self).__init__(config)
        self.layers_name = config['model_config']['layers_name']
        self.layers_size = self.config['model_config']['hidden_size'] #ex 512,256,128, 256, 512
        self.minibatch_size = self.config['data_config']['minibatch_size']
        self.pretrained_models = pretrained_models
        self.pretrained_layers_name = config['model_config']['pretrained_layers']
    def load_pretrained_layers(self, sess):
#         load weights from greedy pretrained layers
#         init saver
        save_params = []
        for layer_name in self.pretrained_layers_name:
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W')
                b = tf.get_variable('b')
                save_params.extend([W, b])
        saver = tf.train.Saver(save_params)
        
        latest_checkpoint = tf.train.latest_checkpoint(self.config['train_config']['pretrained_checkpoint_dir'])
        if latest_checkpoint:
            print("Loading pretrained layers checkpoint {} ...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Pretrained layers loaded")
        
    def load_pretrained_weights(self):
#         load layers train by RBM
#         pretrain model rbms rbm1, rbm2, rbm3
#         layers: inp, h1, h2,h3, h4,h5, out (layers name)
        load_params = []
        for i, model in enumerate(self.pretrained_models):
            #set encoder param
            with tf.variables_scope(self.layers_name[i], reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W')
                b = tf.get_variable('b')
                load_W = tf.assign(W, model.W)
                load_b = tf.assign(b, model.b_h)
                load_params.extend([load_W, load_b])
            with tf.variables_scope(self.layers_name[-i - 1], reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W')
                b = tf.get_variable('b')
                load_W = tf.assign(W, tf.transpose(model.W))
                load_b = tf.assign(b, model.b_v)
                load_params.extend([load_W, load_b])
            
        return load_params
    
    def calculate_l2_loss(self):
        losses = []
        for layer_name in self.layers_name:
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W')
                l2_loss = tf.nn.l2_loss(W)
                losses.append(l2_loss)
        return tf.reduce_sum(losses)
    
    def build_model(self):
        self.initialize_parameters()
        if self.pretrained_models != None:
            self.load_params = self.load_pretrained_weights()
        self.X_train, self.X_train_res = self.create_placeholders()
    #     self.X_cv, self.X_cv_res = self.create_placeholders()
        self.X_test, self.X_test_res = self.create_placeholders()

        # network architecture
        self.X_train_res = self.reconstruct(self.X_train, dropout_prob = self.config['model_config']['dropout_prob'])

        l2_regularize_term = self.calculate_l2_loss()
        self.train_cost = self.calculate_loss(self.X_train, self.X_train_res)
        self.train_cost = self.train_cost + self.config['model_config']['lambda'] * l2_regularize_term
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                    self.config['model_config']['momentum']).minimize(self.train_cost)
    #     # cross-validation
    #     self.X_cv_res = self.cd1(self.X_cv, is_training=False)
    #     self.cv_cost = self.calculate_loss(self.X_cv, self.X_cv_res)
        # test
        self.X_test_res = self.reconstruct(self.X_test, dropout_prob=0)
        self.test_cost = self.calculate_loss(self.X_test, self.X_test_res)



    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        save_params = [self.global_step_tensor]
        for layer_name in self.layers_name:
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W')
                b = tf.get_variable('b')
                save_params.extend([W, b])
        self.saver = tf.train.Saver(save_params)

    
    def create_placeholders(self):
        X = tf.placeholder(tf.float32, shape = (None, self.layers_size[0]), name='X')
        X_res = tf.placeholder(tf.float32, shape = (None, self.layers_size[-1]), name='X_res')
        return X,X_res
  
    def initialize_parameters(self):
        for i, layer_name in enumerate(self.layers_name):
            trainable = not (layer_name in self.pretrained_layers_name)
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W', shape=[self.layers_size[i], self.layers_size[i+1]], initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
                b = tf.get_variable('b', shape=[1, self.layers_size[i+1]], initializer=tf.zeros_initializer(), trainable=trainable)

# rmse loss
    def calculate_loss(self, y_true, y_pred):
        mask_true = tf.cast(y_true > 0, tf.float32)
        masked_squared_error = mask_true * tf.cast(tf.square((y_true - y_pred)), tf.float32)
        masked_mse = tf.reduce_sum(masked_squared_error) / tf.reduce_sum(mask_true)
        return tf.sqrt(masked_mse)
  
    def reconstruct(self, X, dropout_prob, activation='sigmoid'):
        activation_functions = {'sigmoid': tf.nn.sigmoid,
                               'tanh': tf.nn.tanh}
        A = X
        for layer_name in self.layers_name:
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W')
                b = tf.get_variable('b')
                Z = tf.matmul(A, W) + b
                A = activation_functions[activation](Z)
            
#         if(dropout_prob > 0):
#             A1 = tf.nn.dropout(A1, keep_prob=1 - dropout_prob)
            
        return Z
    def get_all_bias(self):
        biases = []
        for layer in self.layers_name:
            with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
                b = tf.get_variable('b')
                biases.append(b)
        return biases
            
        
   

    




class DeepAETrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(DeepAETrainer, self).__init__(sess, model, data, config,logger)

    def train_epoch(self, epoch_index, lr):
        self.num_iter_per_epoch = int(np.ceil(self.data.n_examples / self.model.minibatch_size))
        loop = range(1,self.num_iter_per_epoch+1)
        train_losses = []
        for batch_index in loop:
            minibatch_cost = self.train_step(int(epoch_index * self.num_iter_per_epoch + batch_index), lr)
            train_losses.append(minibatch_cost)

        train_loss = np.mean(train_losses)
        test_loss = self.test_step()
        if epoch_index % 10 == 0:
            print('Epoch {} train loss {}'.format(epoch_index, train_loss))
            print('Epoch {} test loss {}'.format(epoch_index, test_loss))
        cur_it = self.sess.run(self.model.global_step_tensor)
        self.sess.run(self.model.increment_global_step_tensor)
        summaries_dict = {
            'train_loss': train_loss,
            'test_loss': test_loss,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def set_model_attr(self):
#         append input/output size model(n_features)
        self.model.layers_size.append(self.data.n_features)
        self.model.layers_size.insert(0, self.data.n_features)
        
    def save_model(self):
        self.model.save(self.sess)
    
    def train_step(self, iter_index, lr):
        X_train = self.sess.run(self.data.next_batch)
        
        feed_dict_train = {
            self.model.X_train: X_train,
            self.model.learning_rate: lr
        }
        
        _, minibatch_cost = self.sess.run([self.model.optimizer, self.model.train_cost], feed_dict=feed_dict_train)
        return minibatch_cost
      
    def test_step(self):
        X_test = self.data.test
        feed_dict_test = {
            self.model.X_test: X_test
        }
        test_cost = self.sess.run(self.model.test_cost, feed_dict=feed_dict_test)
        return test_cost
    
    def test_load_pretrain(self, sess):
        biases = self.model.get_all_bias()
        for bias in biases:
            b = sess.run(bias)
            print(b[0])




class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        model_description = '{}_deep_ae_batch{}_hiddens{}_lambda{}_learningrate{}_momentum{}_{}'.format(
            config['data_config']['dataset_name'] ,config['data_config']['minibatch_size'], config['model_config']['hidden_size'],
                config['model_config']['lambda'], config['model_config']['learning_rate'], config['model_config']['momentum'], time.time())
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config['train_config']['summary_dir'], model_description),
                                                          self.sess.graph)
#     def summarize_histogram(self, step, summary, scope="init_var"):
#         summary_writer = self.train_summary_writer
#         with tf.variable_scope(scope):
#             summary_writer.add_summary(summary, step)

    # it can summarize scalars and images.
    def summarize(self, step, scope="metrics", summaries_dict=None, mode='scalar'):
        summary_writer = self.train_summary_writer
        with tf.variable_scope(scope):
            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                        if mode == 'scalar':
                            if len(value.shape) <= 1:
                                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                            else:
                                self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])
                        else:
                            if len(value.shape) <= 1:
                                self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])
                            else:
                                self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()
                
    def view_board(self):
      # install tensorboard in colab
      # You can change the directory name
#         LOG_DIR = os.path.join(self.config['train_config']['summary_dir'], "train")
        LOG_DIR = self.config['train_config']['summary_dir']
        if not os.path.exists('ngrok-stable-linux-amd64.zip'):
            get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
        if not os.path.exists('ngrok'):
            get_ipython().system('unzip ngrok-stable-linux-amd64.zip')
        print("Link to view tensorboard")
        if os.path.exists(LOG_DIR):
            get_ipython().system_raw(
            'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
            .format(LOG_DIR))

            get_ipython().system_raw('./ngrok http 6006 &')

            get_ipython().system('curl -s http://localhost:4040/api/tunnels | python3 -c             "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')




# train layer1
config = {
    'train_config':{
        'checkpoint_dir':'/tmp/layer1',
        'num_epochs': 300,
        'summary_dir': '/tmp'
    },
    'data_config':{
        'dataset_name': 'ml-1m',
        'minibatch_size':32,
        'file_path': '../input/ml10m/ratings.dat',
        'seperator': '::'
    },
    'model_config':{
        'hidden_size': [500],
        'layers_name': ['hidden1', 'output'],
        'pretrained_layers': [],
        'dropout_prob': 0,
        'lambda': 0,
        'learning_rate': 0.03,
        'momentum': 0.9,
#         'name_scope': 'deep_ae'
    }
}

tf.reset_default_graph()

# create tensorflow session
with tf.Session() as sess:

  # create your data generator
    data = DataProvider(config)

  # create an instance of the model you want
    model = DeepAutoEncoder(config)
  # create tensorboard logger
    logger = Logger(sess, config)
  # create trainer and pass all the previous components to it
    trainer = DeepAETrainer(sess, model, data, config, logger)
    model.init_saver()
  #load model if exists
    model.load(sess)
  # here you train your model
    trainer.train()




config = {
    'train_config':{
        'pretrained_checkpoint_dir':'/tmp/layer1',
        'checkpoint_dir':'/tmp/layer2',
        'num_epochs': 300,
        'summary_dir': '/tmp'
    },
    'data_config':{
        'dataset_name': 'ml-1m',
        'minibatch_size':32,
        'file_path': '../input/ml10m/ratings.dat',
        'seperator': '::'
    },
    'model_config':{
        'hidden_size': [500, 250],
        'layers_name': ['hidden1', 'hidden2', 'output'],
        'pretrained_layers': ['hidden1'],
        'dropout_prob': 0,
        'lambda': 0,
        'learning_rate': 0.09,
        'momentum': 0.9,
    }
}

tf.reset_default_graph()

# create tensorflow session
with tf.Session() as sess:

  # create your data generator
    data = DataProvider(config)

  # create an instance of the model you want
    model = DeepAutoEncoder(config)
  # create tensorboard logger
    logger = Logger(sess, config)
  # create trainer and pass all the previous components to it
    trainer = DeepAETrainer(sess, model, data, config, logger)
    model.load_pretrained_layers(sess)
    model.init_saver()
  #load model if exists
#     model.load(sess)
  # here you train your model
#     trainer.test_load_pretrain(sess)
    trainer.train()




config = {
    'train_config':{
        'pretrained_checkpoint_dir':'/tmp/layer2',
        'checkpoint_dir':'/tmp/layer3',
        'num_epochs': 300,
        'summary_dir': '/tmp'
    },
    'data_config':{
        'dataset_name': 'ml-1m',
        'minibatch_size':32,
        'file_path': '../input/ml10m/ratings.dat',
        'seperator': '::'
    },
    'model_config':{
        'hidden_size': [500, 250, 500],
        'layers_name': ['hidden1', 'hidden2', 'hidden3', 'output'],
        'pretrained_layers': ['hidden1', 'hidden2'],
        'dropout_prob': 0,
        'lambda': 0,
        'learning_rate': 0.09,
        'momentum': 0.9,
    }
}

tf.reset_default_graph()

# create tensorflow session
with tf.Session() as sess:

  # create your data generator
    data = DataProvider(config)

  # create an instance of the model you want
    model = DeepAutoEncoder(config)
  # create tensorboard logger
    logger = Logger(sess, config)
  # create trainer and pass all the previous components to it
    trainer = DeepAETrainer(sess, model, data, config, logger)
    model.load_pretrained_layers(sess)
    model.init_saver()
  #load model if exists
#     model.load(sess)
  # here you train your model
#     trainer.test_load_pretrain(sess)
    trainer.train()




config = {
    'train_config':{
        'pretrained_checkpoint_dir':'/tmp/layer3',
        'checkpoint_dir':'/tmp/output',
        'num_epochs': 300,
        'summary_dir': '/tmp'
    },
    'data_config':{
        'dataset_name': 'ml-1m',
        'minibatch_size':32,
        'file_path': '../input/ml10m/ratings.dat',
        'seperator': '::'
    },
    'model_config':{
        'hidden_size': [500, 250, 500],
        'layers_name': ['hidden1', 'hidden2', 'hidden3', 'output'],
        'pretrained_layers': ['hidden1', 'hidden2', 'hidden3'],
        'dropout_prob': 0,
        'lambda': 0,
        'learning_rate': 0.09,
        'momentum': 0.9,
    }
}

tf.reset_default_graph()

# create tensorflow session
with tf.Session() as sess:

  # create your data generator
    data = DataProvider(config)

  # create an instance of the model you want
    model = DeepAutoEncoder(config)
  # create tensorboard logger
    logger = Logger(sess, config)
  # create trainer and pass all the previous components to it
    trainer = DeepAETrainer(sess, model, data, config, logger)
    model.load_pretrained_layers(sess)
    model.init_saver()
  #load model if exists
#     model.load(sess)
  # here you train your model
#     trainer.test_load_pretrain(sess)
    trainer.train()




config = {
    'train_config':{
        'checkpoint_dir':'/tmp/output',
        'num_epochs': 500,
        'summary_dir': '/tmp'
    },
    'data_config':{
        'dataset_name': 'ml-1m',
        'minibatch_size':32,
        'file_path': '../input/ml10m/ratings.dat',
        'seperator': '::'
    },
    'model_config':{
        'hidden_size': [500, 250, 500],
        'layers_name': ['hidden1', 'hidden2', 'hidden3', 'output'],
        'pretrained_layers': [],
        'dropout_prob': 0,
        'lambda': 0.00004,
        'learning_rate': 0.01,
        'momentum': 0.9,
    }
}

tf.reset_default_graph()

# create tensorflow session
with tf.Session() as sess:

  # create your data generator
    data = DataProvider(config)

  # create an instance of the model you want
    model = DeepAutoEncoder(config)
  # create tensorboard logger
    logger = Logger(sess, config)
  # create trainer and pass all the previous components to it
    trainer = DeepAETrainer(sess, model, data, config, logger)
#     model.load_pretrained_layers(sess)
    model.init_saver()
  #load model if exists
    model.load(sess)
  # here you train your model
#     trainer.test_load_pretrain(sess)
    trainer.train()




config = {
    'train_config':{
        'checkpoint_dir':'/tmp/deep_ae',
        'num_epochs': 500,
        'summary_dir': '/tmp'
    },
    'data_config':{
        'dataset_name': 'ml-1m',
        'minibatch_size':32,
        'file_path': '../input/ml10m/ratings.dat',
        'seperator': '::'
    },
    'model_config':{
        'hidden_size': [500, 250, 500],
        'layers_name': ['hidden1', 'hidden2', 'hidden3', 'output'],
        'pretrained_layers': [],
        'dropout_prob': 0,
        'lambda': 0.00001,
        'learning_rate': 0.06,
        'momentum': 0.9,
    }
}

tf.reset_default_graph()

# create tensorflow session
with tf.Session() as sess:

  # create your data generator
    data = DataProvider(config)

  # create an instance of the model you want
    model = DeepAutoEncoder(config)
  # create tensorboard logger
    logger = Logger(sess, config)
  # create trainer and pass all the previous components to it
    trainer = DeepAETrainer(sess, model, data, config, logger)
#     model.load_pretrained_layers(sess)
    model.init_saver()
  #load model if exists
#     model.load(sess)
  # here you train your model
#     trainer.test_load_pretrain(sess)
    trainer.train()






