#!/usr/bin/env python
# coding: utf-8



import numpy as np
import scipy.special as scs
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')




class NeuralNet:
   #init the NeuralNet
   def __init__(self, innodes, hiddennodes, outnodes, learningrate):
       #init nodes
       self.inodes = innodes
       self.hnodes = hiddennodes
       self.onodes = outnodes
       #init Learning Rate
       self.learningrate = learningrate
       #linking weight matrix
       self.wi2h = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)) #input to hidden layer weight
       self.wh2o = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)) #hidden to output layer weight
       #activation lambda function
       self.activation_func = lambda x: scs.expit(x)
       pass
   def train(self, inputs_list, targets_list):
       #convert lists
       inputs = np.array(inputs_list, ndmin=2).T
       targets = np.array(targets_list, ndmin=2).T
       #signals into hidden layer
       hidden_inputs = np.dot(inputs, self.wi2h)
       hidden_outputs = self.activation_func(hidden_inputs)
       #signals into output layer
       out_inputs = np.dot(hidden_outputs, self.wh2o)
       final_outputs = self.activation_func(out_inputs)
       #errors = (targets - final outputs)
       output_errors = targets - final_outputs
       hidden_errors = np.dot(self.wh2o.T, output_errors)
       #update weights - the main point
       self.wh2o += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
       self.wi2h += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
       pass
   def query(self, inputs_list):
       #convert list to 2d array
       inputs = np.array(inputs_list, ndmin=2).T
       hidden_inputs = np.dot(self.wi2h, inputs) 
       hidden_outputs = self.activation_func(hidden_inputs)
       out_inputs = np.dot(self.wh2o, hidden_outputs)
       final_outputs = self.activation_func(out_inputs)
       return final_outputs




# number of input, hidden and output nodes 
input_nodes = 784 
hidden_nodes = 100 
output_nodes = 10 
 
# learning rate is 0.3 
learning_rate = 0.3 
 
# create instance of neural network 
n = NeuralNet(input_nodes,hidden_nodes,output_nodes, learning_rate)




labeled_images = open('../input/train.csv', 'r')
training_data_list = labeled_images.readlines() 
labeled_images.close()
for record in training_data_list[1:]: 
    # split the record by the ',' commas 
    all_values = record.split(',') 
    # scale and shift the inputs 
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)




test_data_file = open("../input/test.csv", 'r') 
test_data_list = test_data_file.readlines() 
test_data_file.close()
test_data_list
all_values = test_data_list[1].split(',')




plt.hist(train_images.iloc[i])

