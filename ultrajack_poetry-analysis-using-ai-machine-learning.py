#!/usr/bin/env python
# coding: utf-8




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




data=pd.read_csv("../input/all.csv",index_col=None )




data=data[data.content.str.contains("Copyright")==False]




# check whether there is any null data 
data[data.content.isnull()==True].index.tolist()




# This is the fault data checked by my eye in csv file, so remove it  
data=data[data.content.str.contains("from Dana, 1904")==False]




import nltk




def rhyme(inp, level):
    entries = nltk.corpus.cmudict.entries()
    syllables = [(word, syl) for word, syl in entries if word == inp]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return set(rhymes)




def doTheyRhyme ( word1, word2 ):
    # first, we don't want to report 'glue' and 'unglue' as rhyming words
    # those kind of rhymes are LAME
    if word1.find ( word2 ) == len(word1) - len ( word2 ):
        return False
    if word2.find ( word1 ) == len ( word2 ) - len ( word1 ): 
        return False

    return word1 in rhyme ( word2, 1 )




# First poem in our csv file
data.content[0]




lin=data.content[0].splitlines()
for li in lin:
    print(li)





lin=(data.content[0]).replace(","," ").replace("."," ").replace(";"," ").replace(":"," ").replace("!"," ")
line=lin.splitlines()
i=0
Set=[]
Sentence_set=[]
result=""
print ("---last word of the sentence, see the below for the result ---")
for li in line:
    #print(li)
    Sentence_set.append(li)
    sp=li.split()
    if ("".join(sp[-1:]) is None or  len("".join(sp[-1:])) == 0 or "".join(sp[-1:])==" "):
        continue
    print (str(i)+" "+"".join(sp[-1:]))
    Set.append("".join(sp[-1:]) )
    if(len(Set)%4==0):
        if(  doTheyRhyme(Set[len(Set)-4], Set[len(Set)-3] )==True and  doTheyRhyme(Set[len(Set)-3], Set[len(Set)-2] )==True and doTheyRhyme(Set[len(Set)-2], Set[len(Set)-1] )==True):
            result= ("--AAAA--");
        elif (  doTheyRhyme(Set[len(Set)-4], Set[len(Set)-3] )==True and  doTheyRhyme(Set[len(Set)-3], Set[len(Set)-2] )==False and doTheyRhyme(Set[len(Set)-2], Set[len(Set)-1] )==True):
            result =("--AABB--");
        elif (  doTheyRhyme(Set[len(Set)-4], Set[len(Set)-2] )==True and  doTheyRhyme(Set[len(Set)-3], Set[len(Set)-1] )==False and doTheyRhyme(Set[len(Set)-3], Set[len(Set)-2] )==False):
            result =("--ABAB--");
        elif (  doTheyRhyme(Set[len(Set)-4], Set[len(Set)-1] )==True and  doTheyRhyme(Set[len(Set)-3], Set[len(Set)-1] )==False and doTheyRhyme(Set[len(Set)-3], Set[len(Set)-2] )==True):
            result =("--ABBA--");
    i=i+1
if(result==""):    
    result= ("--No Rhyme--");
print(result)
    




print ("--sentences for similes  in first poem--")
for sen in Sentence_set:
    sen_break = sen.split();
    if (( "like"  in sen_break) or ("as"  in sen_break)):
        print (sen)




print ("--sentences for alliterations  in first poem--")
for sen in Sentence_set:
    sen_break = sen.split();
    if (sen_break is None or  len(sen_break) == 0 or sen_break==" "):
        continue
    counter=1
    threeorfour=1
    last_character=''
    for ele_in in sen_break:
        if (ele_in[0]==last_character):
            threeorfour+=1
            if(counter==3):
                print (" ".join(sen_break) )
        if(counter==1):
            last_character=ele_in[0]
        counter+=1




print ("--sentences for repetitions   in first poem--")
for sen in Sentence_set:
    sen_break = sen.split();
    if (sen_break is None or  len(sen_break) == 0 or sen_break==" "):
        continue
    for ele_in in sen_break:
        sen_break.remove(ele_in)
        if (ele_in in sen_break):
                print (sen +"      -word of Repetitions:"+ ele_in)
        




data.content.str.lower()




data.content=data.content.str.replace('\n', " ")
data.content=data.content.str.replace("\t", " ")
data.content=data.content.str.replace("\r", " ")
data.content=data.content.str.replace(","," ").replace("."," ")




# remove list in the document
remove_list=["A",
"An",
"The",
"Aboard",
"About",
"Above",
"Absent",
"Across",
"After",
"Against",
"Along",
"Alongside",
"Amid",
"Among",
"Amongst",
"Anti",
"Around",
"As",
"At",
"Before",
"Behind",
"Below",
"Beneath",
"Beside",
"Besides",
"Between",
"Beyond",
"But",
"By",
"Circa",
"Concerning",
"Considering",
"Despite",
"Down",
"During",
"Except",
"Excepting",
"Excluding",
"Failing",
"Following",
"For",
"From",
"Given",
"In",
"Inside",
"Into",
"Like",
"Minus",
"Near",
"Of",
"Off",
"On",
"Onto",
"Opposite",
"Outside",
"Over",
"Past",
"Per",
"Plus",
"Regarding",
"Round",
"Save",
"Since",
"Than",
"Through",
"To",
"Toward",
"Towards",
"Under",
"Underneath",
"Unlike",
"Until",
"Up",
"Upon",
"Versus",
"Via",
"With",
"Within",
"Without"]




# replace those words with space
for  value in remove_list:
    data.content=data.content.str.replace(value," ")
data.content




import re
# regular expression, using stemming: try to replace tail of words like ies to y 




data.content = data.content.str.replace("ing( |$)", " ")
data.content = data.content.str.replace("[^a-zA-Z]", " ")
data.content = data.content.str.replace("ies( |$)", "y ")




from sklearn.feature_extraction.text import TfidfVectorizer




vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, analyzer= 'word')




data.head()




data[["content","author","poem name"]]




from sklearn.model_selection import train_test_split




# if there is any empty data, drop it
data.dropna(inplace=True)





data_content_train,data_content_test, data_train_label,data_test_label =train_test_split(data[["content","author","poem name"]],data.type,test_size = 0.2, random_state = 1)




data_test_label_for_age=data.ix[data_test_label.index].age
data_train_label_for_age=data.ix[data_train_label.index].age




data_content_train




train_ = vectorizer.fit_transform(data_content_train.content.as_matrix())
feature_names =vectorizer.get_feature_names()
feature_names
test_ = vectorizer.transform(data_content_test.content.as_matrix())




# check if there is any empty poem name in the file
removelist=data_content_train["poem name"].index[data_content_train["poem name"].isnull()==True].tolist()
removelist




from sklearn import preprocessing
label_au = preprocessing.LabelEncoder()
label_author=label_au.fit_transform(data_content_train.author.as_matrix())
label_authorT=label_au.fit_transform(data_content_test.author.as_matrix())

label_poe_name =TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')  
label_poena=label_poe_name.fit_transform(data_content_train["poem name"].as_matrix())
label_poenaT  =label_poe_name.fit_transform(data_content_test["poem name"].as_matrix())




label_author=np.reshape(label_author, (label_author.shape[0], 1))
label_authorT=np.reshape(label_authorT, (label_authorT.shape[0], 1))




from numpy import array




# We try to catch more feature, but it did not make big difference of result
from sklearn.feature_selection import SelectKBest ,chi2
#y = np.array(data_content_train)
ch2 = SelectKBest(chi2, k=2000)
#X_train=ch2.fit_transform(train_, data_train_label.tolist() )
#X_test = ch2.transform(test_)
X_train=train_;
X_test=test_;




# It did not make big difference of result if we use dense matrix 
#import scipy.sparse as sp
#if(sp.issparse(X_train)==True):
#   X_train = X_train.todense()
#   X_test = X_test.todense()
    




import xgboost as xgb




xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    "eval_metric": 'logloss',
    'silent': 1
}




xgb_params_age = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    "eval_metric": 'error',
    'silent': 1
}




from sklearn import preprocessing
le = preprocessing.LabelEncoder()
a=le.fit_transform(data_train_label.as_matrix())




le2 = preprocessing.LabelEncoder()
a_age=le2.fit_transform(data_train_label_for_age.as_matrix())




dtrain = xgb.DMatrix(X_train, a )
dtest = xgb.DMatrix(X_test)
dtrain_age = xgb.DMatrix(X_train, a_age )
dtest_age = xgb.DMatrix(X_test)




num_boost_rounds = 422
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
result =model.predict(dtest)




num_boost_rounds = 422
model_age = xgb.train(dict(xgb_params_age, silent=0), dtrain_age, num_boost_round=num_boost_rounds)
result_age =model.predict(dtest_age)




result_age
# we need to do more to convert it into our label, which will be renaissance and modern




presult=pd.DataFrame(result)
presult_age=pd.DataFrame(result_age)




presult[(presult.values >= 0.5) & (presult.values < 1.5) ]= 1;
presult[(presult.values >= 1.5) & (presult.values < 2.5) ]=2;
presult[(presult.values >= -0.5) & (presult.values < 0.5) ]=0;

presult_age[(presult_age.values >= -0.5) & (presult_age.values < 0.5) ]=0;
presult_age[(presult_age.values >= 0.5) & (presult_age.values < 1.5) ]= 1;




presult=presult.astype(int)
presult_age=presult_age.astype(int)




result_back=le.inverse_transform(presult.values)
result_back_age=le2.inverse_transform(presult_age.values)




# after conversion 
result_back_age.ravel()




# accuracy for target type 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(data_test_label, result_back)
accuracy




# accuracy for target age
accuracy_age = accuracy_score(data_test_label_for_age, result_back_age)
accuracy_age




# after conversion 
result_back_age




pd.DataFrame({  'poem name': data_content_test["poem name"],
                'correct_data' : data_test_label_for_age+ " " +data_test_label,
                'predict result' : result_back_age.ravel()+" " +result_back.ravel()
                    })

