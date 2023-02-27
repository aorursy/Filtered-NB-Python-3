#!/usr/bin/env python
# coding: utf-8



import json 
import pandas as pd 
from pandas.io.json import json_normalize #package for flattening json in pandas df

#load json object
with open('../input/raw_nyc_phil.json') as f:
    d = json.load(f)

#lets put the data into a pandas df
#clicking on raw_nyc_phil.json under "Input Files"
#tells us parent node is 'programs'
nycphil = json_normalize(d['programs'])
nycphil.head(3)




works_data = json_normalize(data=d['programs'], record_path='works', 
                            meta=['id', 'orchestra','programID', 'season'])
works_data.head(3)




#flatten concerts column here




soloist_data = json_normalize(data=d['programs'], record_path=['works', 'soloists'], 
                              meta=['id'])
soloist_data.head(3)






