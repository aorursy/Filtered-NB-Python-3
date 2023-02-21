#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




ls ../input/




import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, recall_score, precision_score, auc, f1_score
from scipy import interp
from collections import defaultdict




train = pd.read_csv('../input/train.csv')




train['source_type'] = train['source_type'].apply(lambda x: 'unknown' if pd.isnull(x) else x)
lb = LabelBinarizer()
lb.fit(train['source_type'])
f_source_type = lb.transform(train['source_type'])
f_source_type = pd.DataFrame(f_source_type, columns=lb.classes_)




train['cnt'] = 1




song_pop = train[['song_id', 'cnt']].groupby(by='song_id').sum().reset_index()




song_pop['song_rank'] = song_pop['cnt'].rank(ascending=False, method='min')




songs = pd.read_csv('../input/songs.csv', encoding='utf-8')




artist_rank = pd.merge(train[['song_id', 'cnt']], songs[['song_id', 'artist_name']], how='inner', on='song_id')




artist_rank = artist_rank.groupby(by='artist_name').sum().reset_index()
artist_rank['artist_rank'] = artist_rank['cnt'].rank(ascending=False, method='min')




artist_rank = pd.merge(songs[['song_id', 'artist_name']], artist_rank[['artist_name', 'artist_rank']], on='artist_name', how='inner')




members = pd.read_csv('../input/members.csv')
members['registration_init_time'] = members['registration_init_time'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
members['expiration_date'] = members['expiration_date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
members['lifespan_days'] = (members['expiration_date'] - members['registration_init_time'])/np.timedelta64(1, 'D')




trial_users = members['lifespan_days'] <= 9
advance_payers = members['expiration_date'] > datetime(2017,4,1)




members['trial_users'] = 0
members.loc[trial_users, 'trial_users'] = 1




members['advance_payers'] = 0
members.loc[advance_payers, 'advance_payers'] = 1




train_cleaned = pd.concat((train, f_source_type), axis=1)




train_cleaned = pd.merge(train_cleaned, song_pop[['song_id', 'song_rank']], how='left', on='song_id')




train_cleaned = pd.merge(train_cleaned, artist_rank[['song_id', 'artist_rank']], how='left', on='song_id')




train_cleaned['artist_rank'] = train_cleaned['artist_rank'].apply(lambda x: 27383 if pd.isnull(x) else x)




train_cleaned = pd.merge(train_cleaned, songs[['song_id', 'song_length', 'genre_ids', 'language']], how='left', on='song_id')




train_cleaned['song_length'] = train_cleaned['song_length'].apply(lambda x: 241812 if pd.isnull(x) else x)




genre_rank = {'465':1, '485':2, '921':3, '1609':4, '444':5}
train_cleaned['language'] = train_cleaned['language'].apply(lambda x: 0 if pd.isnull(x) else x)
train_cleaned['language'] = train_cleaned['language'].astype(int)
train_cleaned['language'] = train_cleaned['language'].astype(str)




lb = LabelBinarizer()
lb.fit(train_cleaned['language'])
f_language = lb.transform(train_cleaned['language'])




f_lan_names = []
for cl in lb.classes_:
    f_lan_names.append('lan_' + cl)
f_language = pd.DataFrame(f_language, columns=f_lan_names)
train_cleaned = pd.concat((train_cleaned, f_language), axis=1)




train_cleaned['top_genre'] = train_cleaned['genre_ids'].apply(lambda x: genre_rank[x] if x in genre_rank else 6)









train_cleaned = pd.merge(train_cleaned, members[['msno', 'lifespan_days', 'trial_users', 'advance_payers']], how='left', on='msno')




label = 'target'
features = [            u'album',
                       u'artist',            u'listen-with',
                u'local-library',         u'local-playlist',
            u'my-daily-playlist',        u'online-playlist',
                        u'radio',                   u'song',
          u'song-based-playlist',    u'top-hits-for-artist',
       u'topic-article-playlist',                u'unknown',
                    u'song_rank',            u'artist_rank',
                  u'song_length',              
                    u'top_genre',
                       u'lan_-1',                  u'lan_0',
                       u'lan_10',                 u'lan_17',
                       u'lan_24',                  u'lan_3',
                       u'lan_31',                 u'lan_38',
                       u'lan_45',                 u'lan_52',
                       u'lan_59',          u'lifespan_days',
                  u'trial_users',          u'advance_payers']




rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced', random_state=0)




def model_metrics(predicted, actual):
    metrics = dict()
    metrics['cma'] = confusion_matrix(actual, predicted)
    metrics['accuracy'] = accuracy_score(actual, predicted)
    metrics['precision'] = precision_score(actual, predicted)
    metrics['recall'] = recall_score(actual, predicted)
    metrics['f1'] = f1_score(actual, predicted)
    return metrics




rfc.fit(train_cleaned[features], train_cleaned[label])




train_cleaned.head()






