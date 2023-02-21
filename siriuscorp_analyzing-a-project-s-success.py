#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dir = '../input/'

from subprocess import check_output
print(check_output(["ls", dir]).decode("utf8"))
import glob

data = pd.DataFrame()
for f in glob.glob((dir+'*.csv')): # all files in the directory that matchs the criteria.
    data = pd.concat([data,pd.read_csv(f)])




data.head()




useless_columns = ["id","url","category_url","igg_image_url","compressed_image_url","card_type",
                   "category_slug","source_url","friend_team_members","friend_contributors"]
data = data.drop(useless_columns, axis = 1)




data.head(20)




import re
def Remove_Non_Numeric(column):
    return re.sub(r"\D", "", column)

data.balance = data.balance.apply(lambda row : Remove_Non_Numeric(row) )
data.collected_percentage = data.collected_percentage.apply(lambda row : Remove_Non_Numeric(row) )
data.head()




data.nearest_five_percent = data.nearest_five_percent.apply(lambda row: float(row)/100)
data.collected_percentage = data.collected_percentage.apply(lambda row: float(row)/100)
data.head()




def Get_Days_Left(time):
    if  "hour" in time:
        return float(Remove_Non_Numeric(time))/24
    elif "day" in time:
        return float(Remove_Non_Numeric(time))
    else:
        return 0.0
   




data.amt_time_left = data.amt_time_left.apply(lambda row: Get_Days_Left(row))
data.head()




def Clean_Funding(column):
    if  "true" in column.lower():
        return 1
    elif "false" in column.lower() :
        return -1
    else:
        return 0
    
data.in_forever_funding = data.in_forever_funding.apply(lambda row: Clean_Funding(str(row)))
data.in_forever_funding.unique()




data.head()




data.balance = data.balance.apply(lambda row: float(row))




def sb_BarPlot(data,label,measure):
    a4_dims = (11.0, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    plot = sns.barplot(y=label, x=measure,ax=ax, data=data,orient="horizontal")
    
sb_BarPlot(data,"category_name","balance")




data.loc[data.category_name == "Audio"].head()




data.loc[data.category_name == "Energy & Green Tech"].head()




data = data.drop_duplicates()
data.shape




sb_BarPlot(data,"category_name","balance")




a4_dims = (11.0, 8.27)
corr = data.corr()
fig, ax = plt.subplots(figsize=a4_dims)
hm = sns.heatmap(corr,annot=True)




data.collected_percentage = data.collected_percentage.apply(lambda row : np.where(row >= 1.0,1,0))




fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(x='collected_percentage', y='category_name', 
            data=data, orient = "horizontal", ax=ax,
            estimator=lambda x: sum(x==1)/len(x)*100)




import re
from nltk.corpus import stopwords
#Taken from bag of words competition.
def clean_text(text):    
    letters_only = re.sub("[^a-zA-Z]", " ",text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words ))   




new_titles = data.title.apply(lambda title: clean_text(str(title)))




data.title = new_titles




from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 




titles = vectorizer.fit_transform(data.title).toarray()
words = vectorizer.get_feature_names()
counts = np.sum(titles, axis=0)
Word_Count = pd.DataFrame({"Word":words,"Count":counts})
Word_Count = Word_Count.sort_values(by = "Count", ascending = False)
Word_Count.head()




a4_dims = (11.0, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
plot = sns.barplot(y="Word", x="Count",ax=ax, data=Word_Count.head(20),
                   orient="horizontal",estimator = sum)




data.head()




#Dropping these columns, although I would like to use title.
succesful_pj = data.collected_percentage.values
columns = ["title","tagline","collected_percentage"]
data = data.drop(columns, axis = 1)




from sklearn import preprocessing
def transform_label(series):
    le = preprocessing.LabelEncoder()
    le.fit(data.category_name)
    return le.transform(data.category_name)
 
data.category_name = transform_label(data.category_name)
data.currency_code = transform_label(data.currency_code)





from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 25)
clf.fit(data,succesful_pj)




importances = clf.feature_importances_
std = np.std([clf.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure(figsize = (14,8))
plt.title("Feature importances")
plt.bar(range(data.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(data.shape[1]), data.columns,rotation = "vertical")
plt.xlim([-1, data.shape[1]])
plt.show()

