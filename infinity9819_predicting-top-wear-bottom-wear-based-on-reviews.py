#!/usr/bin/env python
# coding: utf-8



# Importing Required Variables
import sys,math, copy, time, os
import re
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from scipy.spatial.distance import cosine
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# print(os.listdir("../input/womens-ecommerce-clothing-reviews"))


np.set_printoptions(threshold=np.nan)
# Reading the Data
original_data = clothing_review = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
clothing_review = clothing_review.dropna(subset=['Review Text'])
clothing_review = clothing_review.dropna(subset=['Clothing ID'])
clothing_review = clothing_review.loc[clothing_review['Department Name'].isin(["Tops","Bottoms"])]
grouped_review = clothing_review.groupby(["Clothing ID"])['Review Text'].apply(' ::: '.join).reset_index()


#Getting Keywords
topWear = [ "top","blouse","shirt","upper","dress","torso","tank","sleeve","body","sweater"]
bottomWear = ["pant","jean","slack","skirt","leg","waist","lower","thigh","trouser","flare"]


# positivewords = []
# with open("../input/positive-and-negetive-words/positive_words", 'r') as readfile :
#     temp = readfile.readline().strip()
#     while temp != "" :
#         positivewords.append(temp)
#         temp = readfile.readline().strip()

# negetivewords = []
# with open("../input/positive-and-negetive-words/negetive_words", 'r') as readfile :
#     temp = readfile.readline().strip()
#     while temp != "" :
#         positivewords.append(temp)
#         temp = readfile.readline().strip()

wordnet_lemmatizer = WordNetLemmatizer()

keyWords = topWear + bottomWear# + positivewords + negetivewords

for i in range(len(keyWords)) :
    keyWords[i] = wordnet_lemmatizer.lemmatize(keyWords[i])
    
# Clearing the data from extra characters
data = []
actual_labels = []
for i in range(len(grouped_review["Review Text"])):
    j = grouped_review["Review Text"][i].lower()
    j = re.sub(r'[^A-Za-z ]', '', j)
    data.append(j)
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    actual_labels.append(id_cloth)
    
# Tokenising the data
tokenizer = RegexpTokenizer(r'\w+')
for i in range(len(data)) :
    data[i] = tokenizer.tokenize(data[i])

# Getting the list of stop words
stopWords = list(stopwords.words('english'))
stopWords = [re.sub(r'[^A-Za-z ]', '', j) for j in stopWords]


# Lemmatizing and removing stop words
wordnet_lemmatizer = WordNetLemmatizer()
dataFiltered = []
for each_review in data :
    temp = []
    for word in each_review : 
        if not word in stopWords and word in keyWords:
            temp.append(wordnet_lemmatizer.lemmatize(word))
    dataFiltered.append(temp)


# dataFiltered.append(keyWords)

# Creating the word list

wordList = list(keyWords)
wordList.sort()

number_of_reviews = len(dataFiltered)
wordListIndex = { wordList[i]: i for i in range(len(wordList))}
nDocsPerWord = {i : 0 for i in wordList}

for i in range(len(actual_labels)) :
    if actual_labels[i] == "Tops" :
        actual_labels[i] = 0
    else :
        actual_labels[i] = 1






tf = np.zeros(shape=(number_of_reviews,len(wordList)))

for i in range(len(dataFiltered)):
    this_doc_accounted = []
    for j in dataFiltered[i] :
        print(j)
        if j in topWear :
            tf[i][wordListIndex[j]] = 1
        elif j in bottomWear :
            tf[i][wordListIndex[j]] = -1
        elif j in keyWords :
            tf[i][wordListIndex[j]]
        if not j in this_doc_accounted :
            this_doc_accounted.append(j)
            print(j in nDocsPerWord)
            nDocsPerWord[j] += 1
            
tf_normalized = copy.deepcopy(tf)
tf_normalized = tf_normalized / tf_normalized.max(axis=0)




tfIdf = copy.deepcopy(tf)

for i in range(number_of_reviews) :
    for k in dataFiltered[i]:
        j = wordListIndex[k]
        if tfIdf[i][j] != 0 :
            tfIdf[i][j] = tfIdf[i][j]*math.log(number_of_reviews/nDocsPerWord[wordList[j]])


tfIdf_normalized = copy.deepcopy(tfIdf)
tfIdf_normalized = tfIdf_normalized / tfIdf_normalized.max(axis=0)




temprow = np.zeros(len(wordList))
for i in range(len(temprow)) :
    if wordList[i] in topWear :
        temprow[i] = 1


tfIdf = np.vstack([tfIdf, temprow])
tf = np.vstack([tf, temprow])
tfIdf_normalized = np.vstack([tfIdf_normalized, temprow])
tf_normalized = np.vstack([tf_normalized, temprow])

print(np.isnan(np.min(tfIdf)))
print(np.isnan(np.min(tf)))
print(np.isnan(np.min(tfIdf_normalized)))
print(np.isnan(np.min(tf_normalized)))




# K-means Clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

kmeans_clothing = KMeans(n_clusters=2,random_state=0).fit(tf)
kmeans_centroids = kmeans_clothing.cluster_centers_

kmeans_labels= kmeans_clothing.labels_
top_label = kmeans_labels[-1]
correct = 0
for i in range(len(kmeans_labels) -1 ) :
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    if kmeans_labels[i] == top_label:
        if id_cloth == "Tops" :
            correct += 1
    else :
        if id_cloth == "Bottoms" :
            correct +=1

print (correct/ (len(kmeans_labels) -1))

# Agg Clustering ---------------------------------
from sklearn.cluster import AgglomerativeClustering

print((~tf.any(axis=1)).any())
# temp = np.append(tf, np.ones((len(tf),1)),axis=1)
agg_iris = AgglomerativeClustering(n_clusters= 2,linkage="average",affinity="manhattan").fit(tf)
#Getting labels
agg_labels = agg_iris.labels_

top_label = agg_labels[-1]
correct = 0
for i in range(len(agg_labels) -1 ) :
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    if agg_labels[i] == top_label:
        if id_cloth == "Tops" :
            correct += 1
    else :
        if id_cloth == "Bottoms" :
            correct +=1

print (correct/ (len(agg_labels) -1))

# GMM Clustering ---------------------------------

gm_labels = GaussianMixture(2).fit_predict(tf)

print(gm_labels)
top_label = gm_labels[-1]
correct = 0
for i in range(len(gm_labels) -1 ) :
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    if gm_labels[i] == top_label:
        if id_cloth == "Tops" :
            correct += 1
    else :
        if id_cloth == "Bottoms" :
            correct +=1

print (correct/ (len(gm_labels) -1))





kmeans_clothing = KMeans(n_clusters=2,random_state=0).fit(tfIdf)
kmeans_centroids = kmeans_clothing.cluster_centers_

kmeans_labels= kmeans_clothing.labels_
top_label = kmeans_labels[-1]
correct = 0
for i in range(len(kmeans_labels) -1 ) :
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    if kmeans_labels[i] == top_label:
        if id_cloth == "Tops" :
            correct += 1
    else :
        if id_cloth == "Bottoms" :
            correct += 1

print (correct/ (len(kmeans_labels) -1))

# Agg Clustering ---------------------------------
from sklearn.cluster import AgglomerativeClustering


agg_iris = AgglomerativeClustering(n_clusters= 2,linkage="average",affinity="manhattan").fit(tfIdf)
#Getting labels
agg_labels = agg_iris.labels_

top_label = agg_labels[-1]
correct = 0
for i in range(len(agg_labels) -1 ) :
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    if agg_labels[i] == top_label:
        if id_cloth == "Tops" :
            correct += 1
    else :
        if id_cloth == "Bottoms" :
            correct +=1

print (correct/ (len(agg_labels) -1))




tf_matrix = tf # D x V matrix 
A = tf_matrix.T 

U, s, V = np.linalg.svd(A, full_matrices=1, compute_uv=1)

K =  2 # number of components

A_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), V[:K, :])) # D x V matrix 

docs_rep = np.dot(np.diag(s[:K]), V[:K, :]).T # D x K matrix 
terms_rep = np.dot(U[:,:K], np.diag(s[:K])) # V x K matrix 

# print (A_reduced)
# print (docs_rep)
# print (terms_rep)

key_word_indices = [wordList.index(key_word) for key_word in keyWords] # vocabulary indices 

key_words_rep = terms_rep[key_word_indices,:]     
query_rep = np.sum(key_words_rep, axis = 0)


svd_start = time.time()
query_doc_cos_dist = [cosine(query_rep, doc_rep) for doc_rep in docs_rep]
svd_end = time.time()
query_doc_sort_index = np.argsort(np.array(query_doc_cos_dist))


max_iter = 5
for rank, sort_index in enumerate(query_doc_sort_index):
    print(rank + 1, ") Cosine value : ", float(query_doc_cos_dist[sort_index]) ,"\n", clothing_review["Review Text"].iloc[sort_index],"\n")
    max_iter -= 1
    if max_iter == 0 :
        break

("")





kmeans_clothing = KMeans(n_clusters=2,random_state=0).fit(docs_rep)


kmeans_labels= kmeans_clothing.labels_
top_label = kmeans_labels[-1]
correct = 0
for i in range(len(kmeans_labels) -1 ) :
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    if kmeans_labels[i] == top_label:
        if id_cloth == "Tops" :
            correct += 1
    else :
        if id_cloth == "Bottoms" :
            correct += 1

print (correct/ (len(kmeans_labels) -1))




from sklearn.cluster import AgglomerativeClustering

print(np.isnan(np.min(docs_rep)))


agg_iris = AgglomerativeClustering(n_clusters= 2,linkage="average",affinity="manhattan").fit(docs_rep)


#Getting labels
agg_labels = agg_iris.labels_

top_label = agg_labels[-1]
correct = 0
for i in range(len(agg_labels) -1 ) :
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    if agg_labels[i] == top_label:
        if id_cloth == "Tops" :
            correct += 1
    else :
        if id_cloth == "Bottoms" :
            correct +=1

print (correct/ (len(agg_labels) -1))




gm_labels = GaussianMixture(2).fit_predict(docs_rep)

print(gm_labels)
top_label = gm_labels[-1]
correct = 0
for i in range(len(gm_labels) -1 ) :
    compare = grouped_review.iloc[i]["Clothing ID"]
    temp = clothing_review.loc[clothing_review['Clothing ID'] == compare]
    id_cloth = temp.iloc[0].loc['Department Name']
    if gm_labels[i] == top_label:
        if id_cloth == "Tops" :
            correct += 1
    else :
        if id_cloth == "Bottoms" :
            correct +=1

print (correct/ (len(gm_labels) -1))




import itertools

ii = itertools.count(docs_rep.shape[0])
tree = [{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in agg_iris.children_]

# print(tree)




get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print(len(docs_rep[:,0]),len(actual_labels))
plt.scatter(docs_rep[:,0][:-1], docs_rep[:,1][:-1], c=actual_labels) # all documents 
plt.scatter(docs_rep[:,0][-1], docs_rep[:,1][-1], marker='+', c=agg_labels[-1]) # the query 
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
plt.plot()
plt.scatter(docs_rep[:,0][:-1], docs_rep[:,1][:-1], c=agg_labels[:-1]) # all documents 
plt.scatter(docs_rep[:,0][-1], docs_rep[:,1][-1], marker='+', c=agg_labels[-1]) # the query 
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
plt.scatter(docs_rep[:,0][:-1], docs_rep[:,1][:-1], c=kmeans_labels[:-1]) # all documents 
plt.scatter(docs_rep[:,0][-1], docs_rep[:,1][-1], marker='+', c=kmeans_labels[-1]) # the query 
plt.show()
plt.scatter(docs_rep[:,0][:-1], docs_rep[:,1][:-1], c=gm_labels[:-1]) # all documents 
plt.scatter(docs_rep[:,0][-1], docs_rep[:,1][-1], marker='+', c=gm_labels[-1]) # the query 
plt.show()




all_data = original_data.dropna()

tops = []
for i in range(1,len(agg_labels) -1) :
    if agg_labels[i] == 1 :
        tops.append( clothing_review.iloc[i].loc["Clothing ID"] )


top_data_pandas = all_data[all_data['Clothing ID'].isin(tops)]
top_data_pandas = top_data_pandas.reset_index(drop=True)
print(top_data_pandas.shape)

top_data = np.zeros(shape=(top_data_pandas.shape[0],3))
print(top_data.shape)

for  index, row in top_data_pandas.iterrows() :
    top_data[index][0] = int(row["Age"])
    top_data[index][1] = int(row["Rating"])
    top_data[index][2] = int(row["Recommended IND"])


agg_tops = AgglomerativeClustering(n_clusters= 2,linkage="average",affinity="manhattan").fit(top_data)
agg_tops_labels= agg_tops.labels_
print(top_data.shape)




data = []
print(top_data_pandas.shape)
for i in top_data_pandas["Review Text"]:
    j = i.lower()
    j = re.sub(r'[^A-Za-z ]', '', j)
    data.append(j)
    
# Tokenising the data
tokenizer = RegexpTokenizer(r'\w+')
for i in range(len(data)) :
    data[i] = tokenizer.tokenize(data[i])


# Getting the list of stop words
stopWords = list(stopwords.words('english'))
stopWords = [re.sub(r'[^A-Za-z ]', '', j) for j in stopWords]

# Lemmatizing and removing stop words
wordnet_lemmatizer = WordNetLemmatizer()
dataFiltered = []
for each_review in data :
    temp = []
    for word in each_review : 
        if not word in stopWords :
            temp.append(wordnet_lemmatizer.lemmatize(word))
    dataFiltered.append(temp)



# Creating the word list
wordList = np.array(dataFiltered)
wordList = np.hstack(wordList)
wordList = list(set(wordList))
wordList.sort()
number_of_reviews = len(dataFiltered)
wordListIndex = { wordList[i]: i for i in range(len(wordList))}
nDocsPerWord = {i : 0 for i in wordList}

tf_top = np.zeros(shape=(number_of_reviews,len(wordList)))
print(tf_top.shape, top_data.shape)

for i in range(len(dataFiltered)):
    this_doc_accounted = []
    for j in dataFiltered[i] :
        tf_top[i][wordListIndex[j]] += 1
        if not j in this_doc_accounted :
            this_doc_accounted.append(j)
            nDocsPerWord[j] += 1

tfIdf_top = copy.deepcopy(tf_top)

print(tfIdf_top.shape)
for i in range(number_of_reviews) :
    for k in dataFiltered[i]:
        j = wordListIndex[k]
        if tfIdf_top[i][j] != 0 :
            tfIdf_top[i][j] = tfIdf_top[i][j]*math.log(number_of_reviews/nDocsPerWord[wordList[j]])
            
#
print(top_data.shape, tfIdf_top.shape)
top_data = np.concatenate((top_data, tfIdf_top), axis=1)




from sklearn.preprocessing import normalize

top_data = top_data / top_data.max(axis=0)

tf_matrix = top_data # D x V matrix 
A = tf_matrix.T 

U, s, V = np.linalg.svd(A, full_matrices=1, compute_uv=1)

K =  2 # number of components

A_reduced = np.dot(U[:,:K], np.dot(np.diag(s[:K]), V[:K, :])) # D x V matrix 

docs_rep = np.dot(np.diag(s[:K]), V[:K, :]).T # D x K matrix 
terms_rep = np.dot(U[:,:K], np.diag(s[:K])) # V x K matrix 

# print (A_reduced)
# print (docs_rep)
# print (terms_rep)

key_word_indices = [wordList.index(key_word) for key_word in keyWords] # vocabulary indices 

key_words_rep = terms_rep[key_word_indices,:]     
query_rep = np.sum(key_words_rep, axis = 0)

def removeOutliers(x):
    to_ret = []
    for i in x :
        print(i)
        if abs(i[0] - i[1]) < 500:
            print("oh ya")
            to_ret.append(i)
    print (to_ret)
    return np.array(to_ret)
    

# docs_rep = removeOutliers(docs_rep)
print(docs_rep.shape)

agg_tops = AgglomerativeClustering(n_clusters= 2,linkage="average",affinity="manhattan").fit(docs_rep)
new_labels = agg_tops.labels_




print(np.where(new_labels==1))
print(docs_rep[177])




get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.scatter(top_data[:,0], top_data[:,1],c= agg_tops_labels) # all documents 
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x =top_data[:,0]
y =top_data[:,1]
z =top_data[:,2]

ax.scatter(x, y, z, c=agg_tops_labels, marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('Rating')
ax.set_zlabel('Recommended IND')
plt.show()

plt.scatter(docs_rep[:,0][:], docs_rep[:,1][:], c=new_labels[:]) # all documents 
plt.show()


plt.scatter(docs_rep[:,0][:], docs_rep[:,1][:], c=top_data[:,2]) # all documents 
plt.show()

