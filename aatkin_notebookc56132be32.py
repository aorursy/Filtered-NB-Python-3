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




station = pd.read_csv('../input/station.csv',encoding="UTF-8")
# trip = pd.read_csv("../input/trip.csv",encoding="UTF-8")
weather = pd.read_csv("../input/trip.csv",encoding="UTF-8")




import csv

with open('../input/trip.csv','r') as fh1:
    with open('./trip_clean.csv','w') as fh2:
        reader = csv.reader(fh1)
        writer = csv.writer(fh2)
        for idx, row in enumerate(reader):
            if idx == 50793:
                pass
            else:
                writer.writerow(row)


    




trips = pd.read_csv('./trip_clean.csv',encoding='UTF-8')




station['neighborhood'] = station['station_id'].apply(lambda x: x.split('-')[0])




station[station['neighborhood'] == 'BT']




trips['start_hood'] = trips['from_station_id'].apply(lambda x: x.split('-')[0])
trips['end_hood'] = trips['to_station_id'].apply(lambda x: x.split('-')[0])




grp = trips.groupby(['start_hood','end_hood','usertype'],as_index=False).agg({"trip_id": "count"})




(grp[(grp['start_hood'] != grp['end_hood'])] # & (grp['end_hood'] == 'CH')]
.sort_values('trip_id',ascending=False))






