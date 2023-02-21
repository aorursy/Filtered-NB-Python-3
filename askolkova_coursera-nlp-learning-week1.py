#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))




text1 = "Ethics are built right into the ideals and objectives of the United Nations."
len(text1) # how many characters




text2 = text1.split(' ') # to split in words / tokens
len(text2)




# find long words (more then 3 char. long)
[w for w in text2 if len(w)>3]




#find capitalized words
[w for w in text2 if w.istitle()]




#find all words that end with 's'
[w for w in text2 if w.endswith('s')]




text3 = 'To be or not to be'
text4 = text3.split(' ')
text4




len(set(text4)) # to find the number of unique words




len(set(w.lower() for w in text4)) 
# To find the number of unique words, whether they start with capital letter
# or not




text5 = 'ouagadougou'
text6 = text5.split('ou')
text6




'ou'.join(text6)




list(text5) # to get the list of characters




[c for c in text5] # alternative way




text8 = "    A quick brown fox jumped over the lazy dog.   " 




text8.split(' ') # returns not what we want




text9 = text8.strip()
text9




text9.split(' ')




text9.find('o')




text9.rfind('o')




text9.replace('o', 'O')




path = '../input/text-names-dates/yob1880.txt'
f = open(path, 'r')




f.readline()




f.seek(0)
text12 = f.read() # to read entire file
len(text12)




text13 = text12.splitlines()
len(text13)




text14 = '"Ethics are built right into the ideals and objectives of the United Nations" #UNSG @ NY Society for Ethical Culture bit.ly/2guVelr @UN @UN_Women  '




text15 = text14.split(' ')
text15




import re




[w for w in text15 if re.search('@[A-Za-z0-9_]+', w)] # getting collouts




re.findall(r'[aieou]', text5)




re.findall(r'[^aeiou]', text5) # find all consonants




#re.findall(r'[\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text_name)




#re.findall(r'\d{1,2} (?:Jan|Feb|Mar) \d{2,4}', text_name)




#re.findall(r'\d{1,2} (?:Jan|Feb|Mar) [a-z]* \d{2,4} ', text_name)




#re.findall(r'(?:\d{1,2} )? (?:Jan|Feb|Mar) [a-z]* (?:\d{1,2} )? \d{2,4} ', text_name)




import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
df




df['text'].str.len()




df['text'].str.split().str.len() # to find the number of tokens




df['text'].str.contains('appointment')




df['text'].str.count(r'\d') # how many times a digit occurs




df['text'].str.findall(r'\d')




# group and find the hours and minutes
df['text'].str.findall(r'(\d?\d):(\d\d)')




# replace weekdays with '???'
df['text'].str.replace(r'\w+day\b', '???')




text_uni = "Résumé"
print(len(text_uni))
print(text_uni)




# if it was in Python 2
text_uni2 = u"Résumé"
print(len(text_uni2))

