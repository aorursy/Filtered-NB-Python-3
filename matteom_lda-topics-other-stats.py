# Author: Matteo Montanari <matteo.montanari25@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

# Creation of iterable function
def articles_in_csv():
    a = open('../input/Reviews.csv', 'rt')
    for row in csv.reader(a, delimiter=',', quotechar='"' , dialect='excel'):
        yield row[9]

n_features = 1000
n_topics = 10
n_top_words = 25

t0 = time()
print("Loading dataset and extracting TF-IDF features...")

# We don't consider the English stop words and words that cover more than 95% of the texts
vectorizer = TfidfVectorizer(max_df=0.95, 
                             min_df=2, 
                             max_features=n_features,
                             stop_words='english')

tfidf = vectorizer.fit_transform(articles_in_csv())
print("done in %0.3fs." % (time() - t0))

# Fit the NMF model
print("Fitting the NMF model with n_features=%d..."
      % ( n_features))
nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
print("done in %0.3fs." % (time() - t0))

feature_names = vectorizer.get_feature_names()

# Print of the results
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

    
# Credits: Olivier Grisel <olivier.grisel@ensta.org>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

con = sqlite3.connect('../input/database.sqlite')

reviewers = pd.read_sql_query('SELECT PROFILENAME, COUNT(PROFILENAME)AS Freq FROM Reviews GROUP BY PROFILENAME ORDER BY(Freq) DESC', con)

print(reviewers)

dimention_of_df =  reviewers.shape
total__reviewers = dimention_of_df[0]

print("\n\n\n")
print("============")
print("This is the number_of_reviewers")
print(total__reviewers)
print("\n")
total_reviews = sum(reviewers['Freq'])
print("TOT number of reviews")
print(total_reviews)
print("============")
print("\n\n\n")


for i in range(5):
    i += 1
    a = i * 1000

    reviews_until_i = sum(reviewers['Freq'][:a])
    print("Number of reviewers taken into acocunt: " + str(a))
    print("Total resulting reviews: " + str(reviews_until_i))
    print("Percentage of reviews   taken into account:  " + str(float(str(reviews_until_i))/float(str(total_reviews))) )
    print("Percentage of reviewers taken into account:  " + str(float(a)/float(total__reviewers)) )
    print("\n")

reviews = reviewers['Freq']

plt.plot(reviews)

plt.axis([-5000, 210000, -10, 460 ])
plt.grid(True)
plt.show()

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

con = sqlite3.connect('../input/database.sqlite')

quantity_list = [100, 200, 300, 500, 
                 1000, 2000, 3000, 4000, 
                 5000, 10000]

# Various amount of reviewers (5.000 rev. are still below the 2.9% of reviewer while 10.000 > 2.9% )
for number_of_reviewers in quantity_list:
    
    query_part_1 = """
            SELECT  ProfileName, 
                    AVG(HelpfulnessNumerator/HelpfulnessDenominator)AS HelpRatio, 
                    AVG(Score)AS AvgScor, 
                    COUNT(PROFILENAME) Freq
            FROM Reviews
            WHERE Reviews.PROFILENAME IN (SELECT PrfName    -- Creation of the list of the most productive 5000 Reviewers
                                          FROM (SELECT  PROFILENAME As PrfName, COUNT(PROFILENAME)AS Freq
                                                                        FROM Reviews  GROUP BY PROFILENAME  ORDER BY(Freq)
                                                                        DESC  )
                                          LIMIT  """ 
    query_part_2 = str(number_of_reviewers)
    
    query_part_3 = """) GROUP BY ProfileName ORDER BY Freq DESC"""
    
    final_query = query_part_1 + query_part_2 + query_part_3
    
    reviewers = pd.read_sql_query(final_query, con)

    # Generation of a corrrelation Matrix between the variables
    # HelpRatio (i.e. the helpfulness ratio)
    # Avarage Score of the post
    # Frequency of posting
    corr_Matrix = reviewers.corr('pearson')
    
    print("The first " + str(number_of_reviewers) + " most writing reviewers \n Correlation Matrix")
    print(corr_Matrix)
    print("\n")

    
# Creating the query valid for ALL the dataset
# Including it for comparison purposes
All_query_part_1 = """
        SELECT  ProfileName, 
                AVG(HelpfulnessNumerator/HelpfulnessDenominator)AS HelpRatio, 
                AVG(Score)AS AvgScor, 
                COUNT(PROFILENAME) Freq
        FROM Reviews
        WHERE Reviews.PROFILENAME IN (SELECT PrfName    -- Creation of the list of the most productive 5000 Reviewers
                                      FROM (SELECT  PROFILENAME As PrfName, COUNT(PROFILENAME)AS Freq
                                                                    FROM Reviews  GROUP BY PROFILENAME  ORDER BY(Freq)
                                                                    DESC  )
                                        """ 

All_query_part_3 = """) GROUP BY ProfileName ORDER BY Freq DESC"""

final_query = All_query_part_1 + All_query_part_3

reviewers = pd.read_sql_query(final_query, con)

corr_Matrix = reviewers.corr('pearson')

print("All reviewers \n Correlation Matrix")
print(corr_Matrix)
print("\n")

# Credits ( ͡° ͜ʖ ͡°) : <https://www.kaggle.com/jasontam>

# These two functions will count the occurencies 
# of the words related to each topic-list
def count_occurrences(word, sentence):
    return sentence.split().count(word)


def tot_topic_occirencies(topic_list, ReviewText):
    n_occ = 0
    for word in topic_list:
        n_occ += count_occurrences(word, ReviewText)
    return n_occ

"""Import of the necessary libraries"""
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

con = sqlite3.connect('../input/database.sqlite')

# This query selects all the articles of the best 2000 writers
# on average the first 200 prolific reviewers write 14 articles
# which means more or less 2800 reviews 
query_ = """
            SELECT  ProfileName, 
                    (HelpfulnessNumerator/HelpfulnessDenominator)AS HelpRatio, 
                    Score, 
                    Text
            FROM Reviews
            WHERE Reviews.ProfileName IN (SELECT PrfName
                                          FROM (SELECT  PROFILENAME As PrfName, COUNT(PROFILENAME)AS Freq
                                                FROM Reviews  GROUP BY PROFILENAME  ORDER BY(Freq) DESC  )
                                          LIMIT (200)) """


reviewers_first_200 = pd.read_sql_query(query_, con)
# reviewers_first_200.head()


# These are the 9 topics we have etracted with the LDA analysis
# we will count the occurencies of the topics for each reviewer
# a DataFrame will be populate with the extimation of the counts

Topic_9 = ["best","free","ve","gluten","tried","far","cookies","mix","bread","sugar","years","brands","used","products","pasta","wheat","make","delicious","use","brand","favorite","tasting","flour","better","rice"]
Topic_8 = ["product","amazon","price","order","buy","store","time","shipping","box","ordered","com","local","stores","grocery","arrived","pack","received","good","purchase","bought","happy","item","www","http","gp"]
Topic_7 = ["great","price","snack","easy","loves","tastes","tasting","healthy","deal","use","taste","value","flavor","recommend","makes","make","highly","works","quick","fast","perfect","low","protein","work","popcorn"]
Topic_6 = ["chocolate","bars","cookies","dark","bar","milk","hot","cookie","cocoa","peanut","butter","sweet","delicious","candy","sugar","rich","protein","snack","granola","box","calories","white","chip","eat","perfect"]
Topic_5 = ["love","dogs","treats","kids","chips","treat","absolutely","just","perfect","favorite","delicious","snack","healthy","flavors","buy","stuff","cats","easy","eat","wish","wonderful","little","yummy","bag","size"]
Topic_4 = ["like","taste","good","flavor","just","really","don","little","water","sweet","tastes","chips","better","try","salt","sugar","drink","think","bit","nice","stuff","make","eat","didn","bad"]
Topic_3 = ["food","dog","cat","treats","dogs","loves","cats","eat","treat","dry","chicken","old","vet","foods","ingredients","feed","pet","healthy","wellness","picky","canned","feeding","small","bag","quality"]
Topic_2 = ["tea","green","teas","drink","bags","cup","black","iced","stash","flavor","drinking","chai","hot","earl","grey","strong","loose","day","honey","leaves","leaf","caffeine","brew","bag","enjoy"]
Topic_1 = ["coffee","cup","strong","cups","bold","roast","keurig","flavor","blend","bitter","smooth","starbucks","dark","morning","coffees","flavored","drink","pods","like","brew","decaf","maker","rich","french","good"]
Topic_0 = ["br","water","ingredients","fat","review","organic","bag","sugar","want","ll","use","make","don","natural","box","oil","serving","stars","people","know","package","way","calories","bit","bottle"]

element_occurencies_topic_0 = []
element_occurencies_topic_1 = []
element_occurencies_topic_2 = []
element_occurencies_topic_3 = []
element_occurencies_topic_4 = []
element_occurencies_topic_5 = []
element_occurencies_topic_6 = []
element_occurencies_topic_7 = []
element_occurencies_topic_8 = []
element_occurencies_topic_9 = []

for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_0.append(tot_topic_occirencies(Topic_0,reviewers_first_200['Text'][ele]))
        
for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_1.append(tot_topic_occirencies(Topic_1,reviewers_first_200['Text'][ele]))

for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_2.append(tot_topic_occirencies(Topic_2,reviewers_first_200['Text'][ele]))

for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_3.append(tot_topic_occirencies(Topic_3,reviewers_first_200['Text'][ele]))

for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_4.append(tot_topic_occirencies(Topic_4,reviewers_first_200['Text'][ele]))

for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_5.append(tot_topic_occirencies(Topic_5,reviewers_first_200['Text'][ele]))

for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_6.append(tot_topic_occirencies(Topic_6,reviewers_first_200['Text'][ele]))

for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_7.append(tot_topic_occirencies(Topic_7,reviewers_first_200['Text'][ele]))

for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_8.append(tot_topic_occirencies(Topic_8,reviewers_first_200['Text'][ele]))
        
for ele in range(len(reviewers_first_200['Text'])):    
     element_occurencies_topic_9.append(tot_topic_occirencies(Topic_9,reviewers_first_200['Text'][ele]))

reviewers_first_200['Topic_0'] = element_occurencies_topic_0
reviewers_first_200['Topic_1'] = element_occurencies_topic_1
reviewers_first_200['Topic_2'] = element_occurencies_topic_2
reviewers_first_200['Topic_3'] = element_occurencies_topic_3
reviewers_first_200['Topic_4'] = element_occurencies_topic_4
reviewers_first_200['Topic_5'] = element_occurencies_topic_5
reviewers_first_200['Topic_6'] = element_occurencies_topic_6
reviewers_first_200['Topic_7'] = element_occurencies_topic_7
reviewers_first_200['Topic_8'] = element_occurencies_topic_8
reviewers_first_200['Topic_9'] = element_occurencies_topic_9

reviewers_first_200_without_text = reviewers_first_200.drop('Text', 1)
reviewers_first_200_without_text_grouped_by_ProfileName = reviewers_first_200_without_text.groupby('ProfileName', as_index=False).mean()





top_10_reviewers_by_HelpRatio = reviewers_first_200_without_text_grouped_by_ProfileName.sort_values('HelpRatio',ascending=False).head(10)
top_10_reviewers_by_SCORE = reviewers_first_200_without_text_grouped_by_ProfileName.sort_values('Score',ascending=False).head(10)

pivot_top_10_reviewers_by_SCORE = pd.melt(top_10_reviewers_by_SCORE, var_name="Topic_Name", id_vars=['ProfileName'], value_vars=['Topic_0','Topic_1','Topic_2','Topic_3','Topic_4','Topic_5','Topic_6','Topic_7','Topic_8','Topic_9'])

A = sns.swarmplot(x='ProfileName', y="value", hue='Topic_Name', data=pivot_top_10_reviewers_by_SCORE, size=10)

sns.set(font_scale=1.3)

plt.legend(bbox_to_anchor=(.9, 1), loc=2, borderaxespad=0.)

for item in A.get_xticklabels():
    item.set_rotation(90)

pivot_top_10_reviewers_by_HelpRatio = pd.melt(top_10_reviewers_by_HelpRatio, var_name="Topic_Name", id_vars=['ProfileName'], value_vars=['Topic_0','Topic_1','Topic_2','Topic_3','Topic_4','Topic_5','Topic_6','Topic_7','Topic_8','Topic_9'])

A = sns.swarmplot(x='ProfileName', y="value", hue='Topic_Name', data=pivot_top_10_reviewers_by_HelpRatio, size=10)

sns.set(font_scale=1.5)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

for item in A.get_xticklabels():
    item.set_rotation(90)
