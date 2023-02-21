#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import pyplot as plt
                                                               
import re                                       # standard Python REGEXP module


from collections import defaultdict             # defaultdict is like a dict with a default
                                                # value generated for missing keys


import numpy as np                              # NUMPY is a popular math package for Python

import csv                                      # standard Python package for comma-separated files,
                                                # very useful for data import 

import nltk                                     # NLTK is Natural Language Tool Kit for Python 




class MultinomialNaiveBayesClassifier(object):
    """Naive Bayes learning model for text classification
    """
    
    def __init__(self, nfeatures, nclasses, reg=1.0):
        """Constructor
        """
        self.nclasses = nclasses 
        self.nfeatures = nfeatures
        
        self.fstat = np.zeros((nclasses, nfeatures))         # Table of word counts for each class.
                                                             # We use data from this table to estimate a
                                                             # probability of a word in a document given
                                                             # the document class.
        
        self.cstat = np.zeros(nclasses)                      # Table of document counts by category.
                                                             # We use data from this table to estimate prior
                                                             # probability distribution of categories in general
                                                             # population of documents
        
        self.reg = reg                                       # Regularization parameter defines a uniform apriory
                                                             # probability distribution for words. 
        
    def update(self, document, class_):
        """Update model by one example
        
        Input:
            document - list of pairs [(word1, count1), (word2, count2), ...]
            class_ - numerical label of document category
        """
        
        for (feature, count) in document:
            self.fstat[class_, feature] += count             # Updates are extremely simple: just update
            self.cstat[class_] += 1                          # the counts using words from a given document 
                                                             # and its category.
    
    def predict(self, document):
        """Compute probabilities of all categories for input document
        
        Input:
            document - list of pairs [(word1, count1), (word2, count2), ...]
        """
        
        # Start with apriory probabilities of categories
        score = np.log(self.cstat + 1)
        
        # Update by a regularized likelihood of each word 
        for (feature, count) in document:
            score += count * np.log(self.fstat[:, feature] + self.reg)                    - count * np.log(np.sum(self.fstat, axis=1) + self.reg*self.nfeatures)
        
        # Transform from logarithms to actual probabilities:
        prob = np.exp(score - np.min(score))                  # substract constant to avoid underflow!
        
        # Normalize probibilities to one
        return prob / np.sum(prob)
        
    def estimates(self):
        """Get normalized estimates of class conditional probabilities
        """
        norm = np.sum(self.fstat, axis=1) + self.reg*self.nfeatures
        return (self.fstat + self.reg) / np.kron(norm, np.ones((self.nfeatures, 1))).transpose()




def load(filename):
    """Load and preprocess dataset from file
    """
    
    stemmer = nltk.stem.PorterStemmer()                             # Stemmer instance
    
    pattern_remove_non_aplanumeric = re.compile('\W+')              # Pattern to remove all non-alphanumeric symbols
    
    documents = []
    categories = defaultdict(int)
    words = defaultdict(int)
    
    with open(filename) as fid:
        reader = csv.reader(fid)                                    # Initialize CSV reader object
        
        header = next(reader)                                  # Read file header line
        
        count = 0
        for line in reader:                                                     # Read dataset line by line:
            category = line[4]                                                  # * category is in 5th column
            document = line[1].strip()                                          # * document line
            document = re.sub(pattern_remove_non_aplanumeric, ' ', document)    # * remove non-alphanumeric symbols
            document = [str.lower(w) for w in document.split()]                 # * split into words by whitespace
            document = [stemmer.stem(w) for w in document]                      # * replace words by stems
            document = [w for w in document if len(w) > 1]                      # * filter single character words
            
            categories[category] += 1                                           # compute categories count
            for w in document:        
                words[w] += 1                                                   # compute words count
            
            documents.append((category, document))                              # add document to output
    
    return categories, words, documents 




# Load data
categories, words, documents = load('../input/uci-news-aggregator.csv')




# Examine statistics of the dataset
print('CATEGORIES: {:d}'.format(len(categories)))
print(' '.join('{}={}'.format(c, categories[c]) for c in sorted(categories, key=categories.get)))
print() 

print('WORDS: {} \n'.format(len(words)))
threshold = 50
word_stats = {j: 0 for j in range(1, threshold + 1)}
for w in words:
    count = words[w]
    count = count if count < 50 else 50
    word_stats[count] += 1
print('WORDS FREQUENCIES:') 
print(' '.join('{}={}'.format(c, word_stats[c]) for c in range(1, threshold + 1)))
print()

top = 50
print ('TOP {} WORDS:'.format(top))
print (' '.join('{}={}'.format(w, words[w]) for w in sorted(words, key=words.get, reverse=True)[:top]))




# Collect classes
classes_ = dict()
class_id = 0
for c in sorted(categories, key=categories.get, reverse=True):
    classes_[c] = class_id
    class_id += 1

# Collect vocabulary, filtering out rare words
vocabulary = dict()
min_frequency = 2
word_id = 0

for w in sorted(words, key=words.get, reverse=True): 
    if words[w] < min_frequency:
        break

    vocabulary[w] = word_id
    word_id += 1

# Encode documents as bag of words from vocabulary
for j, d in enumerate(documents):
    class_ = classes_[d[0]]
    document = [vocabulary[w] for w in d[1] if w in vocabulary]
    unique, counts = np.unique(document, return_counts=True)
    documents[j] = (class_, np.asarray((unique, counts)).T)




# Shuffle data at random
N = len(documents)
rnd = np.random.RandomState(seed=1923)
rnd.shuffle(documents)




# Split dataset into train, validation and test sets
train_set_idx = range(int(N * 0.8))
print ('Train: ', min(train_set_idx), max(train_set_idx))

validation_set_idx = range(int(N * 0.8), int(N * 0.82))
print ('Validation: ', min(validation_set_idx), max(validation_set_idx))

test_set_idx = range(int(N * 0.82), N)
print ('Train: ', min(test_set_idx), max(test_set_idx))




# Define test metrics
def metrics(documents_idx):
    total = 0
    correct = 0
    entropy = 0
    for k in documents_idx:
        total += 1
        (class_, document) = documents[k]
        probabilities = classifier.predict(document)
        prediction = np.argmax(probabilities)
        entropy += - np.log(probabilities[class_])
        if prediction == class_:
            correct += 1

    precision = float(correct) / total
    entropy = entropy / total
    
    return precision, entropy




# Train classifier
classifier = MultinomialNaiveBayesClassifier(nclasses=len(classes_), nfeatures=len(vocabulary))
learn_curves = []

for j in train_set_idx:
    (class_, document) = documents[j]
    classifier.update(document, class_)
    
    if j % 10000 == 0:
        precision, entropy = metrics(validation_set_idx)
        learn_curves.append((j, precision, entropy))
        print ('iter={:d}, precision={:f}, entropy={:f}'.format(j, precision, entropy))
        
print ('Training complete')




# Compute final score
print ('Test: precision={:f}, entropy={:f}'.format(*metrics(test_set_idx)))

