#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install datefinder')




get_ipython().system('pip install geotext')




import pandas as pd
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import svm
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import math
import geopandas as gpd
from geotext import GeoText
import datefinder

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
y=train.target
X_train,X_val,y_train,y_val = train_test_split(train,y,test_size=0.2)    




def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')




print_full(X_train[X_train.target.eq(0)].head(15))




print_full(X_train[X_train.target.eq(1)].head(15))




print_full(X_train.describe(include='object'))




print(X_train.describe(include='number'))




train_part=pd.concat(g for _, g in X_train.groupby("text") if len(g) > 1)
print(train_part.head(15))




print_full(X_train[X_train.keyword.isna()].head(15))




#Ajouter les mots suivants aux listes anthrax, pathogen, pathogens
#print_full(keyword_list)

add={'key':['anthrax','pathogen']}
addendum=pd.DataFrame(add)

keyword_list=pd.DataFrame(train.keyword.unique())
keyword_list.columns=['key']
keyword_list=keyword_list.append(addendum)
keyword_list.dropna(inplace=True)


keyword_list2=pd.DataFrame(test.keyword.unique())
keyword_list2.columns=['key']
keyword_list2=keyword_list2.append(addendum)
keyword_list2 = keyword_list2.dropna(inplace=True)




def keyword(word1, word2, key_list) : 
    for i in range(len(key_list)) : 
        if key_list.iloc[i,0].count("%20")>0 : 
            temp=key_list.iloc[i,0].split("%20")
            if str.lower(word1).find(temp[0])!=-1 and str.lower(word2).find(temp[1])!=-1 :
                return key_list.iloc[i,0]
        else : 
            if word1.find(key_list.iloc[i,0])!=-1 : 
                print("trouvé : "+ word1+" " + key_list.iloc[i,0])
                return key_list.iloc[i,0]
    return ""




X_train['k_list']=""
X_val['k_list']=""
test['k_list']=""
l2=""
for label, row in X_train.iterrows():    
    temp=row['text'].split()
    l=""
    for i in range(len(temp)):
        j=i+1
        if i<len(temp)-1:
            l2=keyword(temp[i], temp[j],keyword_list)
            if l2!="" : 
                l= l +" "+ l2
        else : 
            l2=keyword(temp[i], "",keyword_list)
            if l2!="" : 
                l= l +" " +l2
    X_train.loc[label,'k_list']=l

    
l2=""
for label, row in X_val.iterrows():    
    temp=row['text'].split()
    l=""
    for i in range(len(temp)):
        j=i+1
        if i<len(temp)-1:
            l2=keyword(temp[i], temp[j],keyword_list)
            if l2!="" : 
                l= l +" "+ l2
        else : 
            l2=keyword(temp[i], "",keyword_list)
            if l2!="" : 
                l= l +" " +l2
    X_val.loc[label,'k_list']=l

l2=""
for label, row in test.iterrows():    
    temp=row['text'].split()
    l=""
    for i in range(len(temp)):
        j=i+1
        if i<len(temp)-1:
            l2=keyword(temp[i], temp[j],keyword_list)
            if l2!="" : 
                l= l +" "+ l2
        else : 
            l2=keyword(temp[i], "",keyword_list)
            if l2!="" : 
                l= l +" " +l2
    test.loc[label,'k_list']=l




print_full(X_train[['k_list', 'text']].head(40))




print(Dtrain.k_list.str.count(row['key']).sum())
print(X_train.k_list.str.count(row['key']).sum())




##### créer liste des scores de chacun des mots

Dtrain=X_train[X_train.target.eq(1)]

for label, row in keyword_list.iterrows() : 
    keyword_list.loc[label,'rate']=((Dtrain.k_list.str.count(row['key']).sum())/(X_train.k_list.str.count(row['key']).sum()))-0.5

keyword_list['rate'].fillna(0, inplace=True)


Dtest=test[test.target.eq(1)]

for label, row in keyword_list2.iterrows() : 
    keyword_list2.loc[label,'rate']=((test.k_list.str.count(row['key']).sum())/(test.k_list.str.count(row['key']).sum()))-0.5

keyword_list2['rate'].fillna(0, inplace=True)




print_full(keyword_list.head(100))




for label, row in X_train.iterrows():
    res=0
    temp=row['k_list'].split()
    l=""
    for i in range(len(temp)):
        #print(type(keyword_list.loc[keyword_list['key']==temp[i],['rate']].iloc[0,0]))
        #print(keyword_list.loc[keyword_list['key']==temp[i],['rate']])
        res+=keyword_list.loc[keyword_list['key']==temp[i],['rate']].iloc[0,0]
        X_train.loc[label,'score']=res

X_train['score'].fillna(0, inplace=True)

for label, row in X_val.iterrows():
    res=0
    temp=row['k_list'].split()
    l=""
    for i in range(len(temp)):
        res+=keyword_list.loc[keyword_list['key']==temp[i],['rate']].iloc[0,0]
        X_val.loc[label,'score']=res

X_val['score'].fillna(0, inplace=True)

for label, row in test.iterrows():
    res=0
    temp=row['k_list'].split()
    l=""
    for i in range(len(temp)):
        res+=keyword_list2.loc[keyword_list2['key']==temp[i],['rate']].iloc[0,0]
        test.loc[label,'score']=res

test['score'].fillna(0, inplace=True)




print(X_train.head(15))




def has_keyword(sentence, df_k):
    """"This function returns the keyword listed in df_k that appears in the sentence or "" """
    for i in range(len(df_k)) :
        #print(" : "+df_k.iloc[i,0])
        if sentence.find(df_k.iloc[i,0])!=-1:
            #print("Trouvé")
            return df_k.iloc[i,0]
    return ""

def nb_keyword(sentence, df_k):
    """"This function returns the number of keyword listed in df_k that appears in the sentence or "" """
    result=0
    for i in range(len(df_k)) :
        #print(" : "+df_k.iloc[i,0])
        if sentence.find(df_k.iloc[i,0])!=-1:
            #print("Trouvé")
            result=result+1
    return result
def haslink(sentence):
    return sentence.count("http:")
def hassaut(sentence):
    return sentence.count("\n")

def hasap(sentence):
    return sentence.count("'")
def hasupperletter(sentence):
    return sum(1 for c in sentence if c.isupper())
def hasat(sentence):
    return sentence.count("@")
def hastags(sentence):
    return sentence.count("#")
def hasmoney(sentence):
    result=str.lower(sentence).count("money")
    result=result+str.lower(sentence).count("dollar")
    result=result+sentence.count("$")
    return result
def hasfun(sentence):
    result=str.lower(sentence).count("fun")
    result=result+str.lower(sentence).count("game")
    result=result+str.lower(sentence).count("lol")
    return result
    

def hasdoublekey(sentence):
    return sentence.count("%20")

def hasnumber(sentence):
    l = []
    for t in sentence.split():
        try:
            l.append(float(t))
        except ValueError:
            pass
    return len(l)
def hasreligions(sentence):
    result=str.lower(sentence).count("god")
    result=result+str.lower(sentence).count("temple")
    result=result+str.lower(sentence).count("sikh")
    result=result+str.lower(sentence).count("pray")
    result=result+str.lower(sentence).count("pastor")
    result=result+str.lower(sentence).count("prophet")
    result=result+str.lower(sentence).count("islam")
    result=result+str.lower(sentence).count("muslim")
    result=result+str.lower(sentence).count("mosque")
    return result

def hassex(sentence):
    result=str.lower(sentence).count("sex")
    result=result+str.lower(sentence).count("ass")
    result=result+str.lower(sentence).count("cock")
    return result

def haspunct(sentence):
    result=sentence.count(".")
    result=result+sentence.count("?")
    result=result+sentence.count("!")
    result=result+sentence.count(":")
    result=result+sentence.count(";")
    return result

def haslocations(sentence) : 
    places=GeoText(sentence)
    countries=places.countries
    cities=places.cities
    return len(countries)+len(cities)
def hasphoto(sentence) : 
    result=str.lower(sentence).count("photo")
    result=result+str.lower(sentence).count("video")
    return result
def hasdates(sentence) : 
    #print(sentence)
    result=0
    try : 
        matches = list(datefinder.find_dates(sentence))
        result= len(matches)
    finally : 
        return result
    




print(hasreligions("I love Paris and Manhattan and london and Hong Kong in February 2010 photo Photo photos god"))









#train['nbcar']=train.apply(lambda row: len(row.text), axis=1)
#train['wordcount']=train.apply(lambda row: len(row.text.split()), axis=1)
#train['av_wordlength']=train.apply(lambda row: row.nbcar/row.wordcount, axis=1)
#train['key'] = train.apply(lambda row: 1 if str(row.keyword).strip() and str(row.keyword) else 0,axis=1)
#train['nb_keyword']=train.apply(lambda row: nb_keyword(str.lower(row.text),keyword_list), axis=1)
#train['loc'] = train.apply(lambda row: 1 if not pd.isnull(row.location) else 0,axis=1)
#train['link']=train.apply(lambda row: haslink(row.text),axis=1)
#train['upletter']=train.apply(lambda row: hasupperletter(row.text),axis=1)
#train['ap']=train.apply(lambda row: hasap(row.text),axis=1)
#train['tag']=train.apply(lambda row: hastags(row.text),axis=1)
#train['at']=train.apply(lambda row: hasat(row.text),axis=1)
#train['money']=train.apply(lambda row: hasmoney(row.text),axis=1)
#train['found_loc']=train.apply(lambda row: haslocations(row.text),axis=1)
#train['date']=train.apply(lambda row: hasdates(row.text),axis=1)
#train['dbk']=train.apply(lambda row: 0 if pd.isnull(row.keyword) else hasdoublekey(row.keyword),axis=1)
#train['number']=train.apply(lambda row: hasnumber(row.text),axis=1)
#train['photo']=train.apply(lambda row: hasphoto(row.text),axis=1)



X_train['nbcar']=X_train.apply(lambda row: len(row.text), axis=1)
X_train['wordcount']=X_train.apply(lambda row: len(row.text.split()), axis=1)
X_train['av_wordlength']=X_train.apply(lambda row: row.nbcar/row.wordcount, axis=1)
X_train['key'] = X_train.apply(lambda row: 1 if str(row.keyword).strip() and str(row.keyword) else 0,axis=1)
X_train['nb_keyword']=X_train.apply(lambda row: nb_keyword(str.lower(row.text),keyword_list), axis=1)
X_train['loc'] = X_train.apply(lambda row: 1 if not pd.isnull(row.location) else 0,axis=1)
X_train['link']=X_train.apply(lambda row:  haslink(row.text) ,axis=1)
X_train['upletter']=X_train.apply(lambda row: hasupperletter(row.text),axis=1)
X_train['ap']=X_train.apply(lambda row: hasap(row.text),axis=1)
X_train['tag']=X_train.apply(lambda row: hastags(row.text),axis=1)
X_train['at']=X_train.apply(lambda row: hasat(row.text),axis=1)
X_train['money']=X_train.apply(lambda row: hasmoney(row.text),axis=1)
X_train['found_loc']=X_train.apply(lambda row: haslocations(row.text),axis=1)
X_train['date']=X_train.apply(lambda row: hasdates(row.text),axis=1)
X_train['dbk']=X_train.apply(lambda row: 0 if pd.isnull(row.keyword) else hasdoublekey(row.keyword),axis=1)
X_train['number']=X_train.apply(lambda row: hasnumber(row.text),axis=1)
X_train['photo']=X_train.apply(lambda row: hasphoto(row.text),axis=1)
X_train['religion']=X_train.apply(lambda row: hasreligions(row.text),axis=1)
X_train['fun']=X_train.apply(lambda row: hasfun(row.text),axis=1)
X_train['sex']=X_train.apply(lambda row: hassex(row.text),axis=1)
X_train['saut']=X_train.apply(lambda row: hassaut(row.text),axis=1)
X_train['punct']=X_train.apply(lambda row: haspunct(row.text),axis=1)


X_val['nbcar']=X_val.apply(lambda row: len(row.text), axis=1)
X_val['wordcount']=X_val.apply(lambda row: len(row.text.split()), axis=1)
X_val['av_wordlength']=X_val.apply(lambda row: row.nbcar/row.wordcount, axis=1)
X_val['key'] = X_val.apply(lambda row: 1 if str(row.keyword).strip() and str(row.keyword) else 0,axis=1)
X_val['nb_keyword']=X_val.apply(lambda row: nb_keyword(str.lower(row.text),keyword_list), axis=1)
X_val['loc'] = X_val.apply(lambda row: 1 if not pd.isnull(row.location) else 0,axis=1)
X_val['link']=X_val.apply(lambda row: haslink(row.text) ,axis=1)
X_val['upletter']=X_val.apply(lambda row: hasupperletter(row.text),axis=1)
X_val['ap']=X_val.apply(lambda row: hasap(row.text),axis=1)
X_val['tag']=X_val.apply(lambda row: hastags(row.text),axis=1)
X_val['at']=X_val.apply(lambda row: hasat(row.text),axis=1)
X_val['money']=X_val.apply(lambda row: hasmoney(row.text),axis=1)
X_val['found_loc']=X_val.apply(lambda row: haslocations(row.text),axis=1)
X_val['date']=X_val.apply(lambda row: hasdates(row.text),axis=1)
X_val['dbk']=X_val.apply(lambda row: 0 if pd.isnull(row.keyword) else hasdoublekey(row.keyword),axis=1)
X_val['number']=X_val.apply(lambda row: hasnumber(row.text),axis=1)
X_val['photo']=X_val.apply(lambda row: hasphoto(row.text),axis=1)
X_val['religion']=X_val.apply(lambda row: hasreligions(row.text),axis=1)
X_val['fun']=X_val.apply(lambda row: hasfun(row.text),axis=1)
X_val['sex']=X_val.apply(lambda row: hassex(row.text),axis=1)
X_val['saut']=X_val.apply(lambda row: hassaut(row.text),axis=1)
X_val['punct']=X_val.apply(lambda row: haspunct(row.text),axis=1)


test['nbcar']=test.apply(lambda row: len(row.text), axis=1)
test['wordcount']=test.apply(lambda row: len(row.text.split()), axis=1)
test['av_wordlength']=test.apply(lambda row: row.nbcar/row.wordcount, axis=1)
test['key']=test.apply(lambda row: 1 if str(row.keyword).strip() and str(row.keyword) else 0,axis=1)
test['nb_keyword']=test.apply(lambda row: nb_keyword(str.lower(row.text),keyword_list2), axis=1)
test['loc'] = test.apply(lambda row: 1 if not pd.isnull(row.location) else 0,axis=1)
test['link']=test.apply(lambda row:  haslink(row.text),axis=1)
test['upletter']=test.apply(lambda row: hasupperletter(row.text),axis=1)
test['ap']=test.apply(lambda row: hasap(row.text),axis=1)
test['tag']=test.apply(lambda row: hastags(row.text),axis=1)
test['at']=test.apply(lambda row: hasat(row.text),axis=1)
test['money']=test.apply(lambda row: hasmoney(row.text),axis=1)
test['found_loc']=test.apply(lambda row: haslocations(row.text),axis=1)
test['date']=test.apply(lambda row: hasdates(row.text),axis=1)
test['dbk']=test.apply(lambda row: 0 if pd.isnull(row.keyword) else hasdoublekey(row.keyword),axis=1)
test['number']=test.apply(lambda row: hasnumber(row.text),axis=1)
test['photo']=test.apply(lambda row: hasphoto(row.text),axis=1)
test['religion']=test.apply(lambda row: hasreligions(row.text),axis=1)
test['fun']=test.apply(lambda row: hasfun(row.text),axis=1)
test['sex']=test.apply(lambda row: hassex(row.text),axis=1)
test['saut']=test.apply(lambda row: hassaut(row.text),axis=1)
test['punct']=test.apply(lambda row: haspunct(row.text),axis=1)









sns.set_style("whitegrid")
#print(train[['key','keyword']].head(100).to_string())
#print(train.describe(include='all'))
NDtrain=X_train[X_train.target.eq(0)]
Dtrain=X_train[X_train.target.eq(1)]
sns.distplot(NDtrain['key'],kde=False, label="Non disaster tweet")
sns.distplot(Dtrain['key'],kde=False, label="Disaster tweet")
plt.legend(prop={'size': 12})
plt.xlabel("Presence of a keyword in the tweet")
plt.title("Impact of the keyword presence on tweets nature")
plt.show()




sns.set_context("paper", font_scale=2)
sns.set_style("white")
plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(8,6))
sns.despine(left=True)

B=['Non disaster tweet','Disaster tweet']
ax.hist([NDtrain['nb_keyword'], Dtrain['nb_keyword']],  histtype='bar', 
        align='mid', label=B, alpha=0.4)#, rwidth=0.6)

plt.legend(prop={'size': 12})
plt.xlabel("Number of keywords in a tweet")
plt.title("Impact of the number of keywords on target")
plt.show()




sns.set_context("paper", font_scale=2)
sns.set_style("white")
plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(6,6))
sns.despine(left=True)

B=['Non disaster tweet','Disaster tweet']
ax.hist([NDtrain['link'], Dtrain['link']],  histtype='bar', 
        align='mid', label=B, alpha=0.4)#, rwidth=0.6)

plt.legend(prop={'size': 12})
plt.xlabel("Link filled in a tweet")
plt.title("Impact of the links on target")
plt.show()




sns.set_context("paper", font_scale=2)
sns.set_style("white")
plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(6,6))
sns.despine(left=True)

B=['Non disaster tweet','Disaster tweet']
ax.hist([NDtrain['tag'], Dtrain['tag']],  histtype='bar', 
        align='mid', label=B, alpha=0.4)#, rwidth=0.6)

plt.legend(prop={'size': 12})
plt.xlabel("Tags filled in a tweet")
plt.title("Impact of the tags on target")
plt.show()




sns.set_context("paper", font_scale=2)
sns.set_style("white")
plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(6,6))
sns.despine(left=True)

B=['Non disaster tweet','Disaster tweet']
ax.hist([NDtrain['found_loc'], Dtrain['found_loc']],  histtype='bar', 
        align='mid', label=B, alpha=0.4)#, rwidth=0.6)

plt.legend(prop={'size': 12})
plt.xlabel("Locations in a tweet")
plt.title("Impact of the locations inside a tweet on target")
plt.show()




sns.set_context("paper", font_scale=2)
sns.set_style("white")
plt.rc('text', usetex=False)
fig, ax = plt.subplots(figsize=(6,6))
sns.despine(left=True)

B=['Non disaster tweet','Disaster tweet']
ax.hist([NDtrain['loc'], Dtrain['loc']],  histtype='bar', 
        align='mid', label=B, alpha=0.4)#, rwidth=0.6)

plt.legend(prop={'size': 12})
plt.xlabel("Location filled in a tweet")
plt.title("Impact of the location on target")
plt.show()




print_full(X_train.loc[X_train['loc']==1, ['location', 'target', 'text']].head(40))




fig, ax = plt.subplots(figsize=(6,6))
sns.despine(left=True)

B=['Non disaster tweet','Disaster tweet']
ax.hist([NDtrain['nbcar'], Dtrain['nbcar']],  histtype='bar', 
        align='mid', label=B, alpha=0.4)#, rwidth=0.6)

plt.legend(prop={'size': 12})
plt.xlabel("Number of caracters in a tweet")
plt.title("Impact of the number of caracters on target")
plt.show()




fig, ax = plt.subplots(figsize=(8,6))
sns.despine(left=True)

B=['Non disaster tweet','Disaster tweet']
ax.hist([NDtrain['wordcount'], Dtrain['wordcount']],  histtype='bar', 
        align='mid', label=B, alpha=0.4)#, rwidth=0.6)

plt.legend(prop={'size': 12})
plt.xlabel("Words number in a tweet")
plt.title("Impact of the words number on target")
plt.show()




fig, ax = plt.subplots(figsize=(8,6))
sns.despine(left=True)

B=['Non disaster tweet','Disaster tweet']
ax.hist([NDtrain['av_wordlength'], Dtrain['av_wordlength']],  histtype='bar', 
        align='mid', label=B, alpha=0.4)#, rwidth=0.6)

plt.legend(prop={'size': 12})
plt.xlabel("Words number in a tweet")
plt.title("Impact of the words average length on target")
plt.show()




sns.set_style("whitegrid")
ax=sns.catplot(x="wordcount", y="nbcar", kind="box",hue="target",data=X_train, height=10)
ax.set(xlabel='Number of words', ylabel='Tweet Caracter count')
plt.title("Tweets Caracter count by number of words and Result")
plt.show()




sns.set_style("whitegrid")
ax=sns.catplot(x="nb_keyword", y="nbcar", kind="box",hue="target",data=X_train, height=10)
ax.set(xlabel='Number of keyword', ylabel='Tweet Caracter count')
plt.title("Tweets Caracter count by number of keywords and Result")
plt.show()




sns.set_style("whitegrid")
ax=sns.catplot(x="loc", y="nbcar", kind="box",hue="target",data=X_train, height=10)
ax.set(xlabel='Location filled', ylabel='Tweet Caracter count')
plt.title("Tweets Caracter count by filled location and Result")
plt.show()




#clf=svm.SVC()

clf=RandomForestClassifier()

#Tags addition reduce accuracy :-(
features=['nbcar','wordcount','av_wordlength','nb_keyword','loc', 'link', 'upletter', 'ap', 'tag', 'at', 
          'money',  'found_loc','date', 'dbk', 'number', 'photo', 'religion', 'sex', 'fun', 'saut', 'punct', 'score']
clf.fit(X_train[features],X_train[['target']])
res_val=clf.predict(X_val[features])
X_val['pred']=res_val
erreur_pred=X_val[X_val['pred']!=X_val['target']]
print(clf.score(X_val[features], y_val))
print_full(erreur_pred[['target', 'pred','nb_keyword' ,'link', 'keyword','found_loc','date',  'number' ,'text' ]])









y_test=clf.predict(test[features])
my_submission=pd.DataFrame({'id': test.id, 'target': y_test})
my_submission.to_csv('submission.csv', index=False)

