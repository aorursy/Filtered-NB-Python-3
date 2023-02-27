#!/usr/bin/env python
# coding: utf-8



# modules we'll use
import pandas as pd
import numpy as np

# helpful modules
import fuzzywuzzy
from fuzzywuzzy import process
import chardet

# set seed for reproducibility
np.random.seed(0)




# look at the first ten thousand bytes to guess the character encoding
with open("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))

# check what the character encoding might be
print(result)




# read in our dat
suicide_attacks = pd.read_csv("../input/PakistanSuicideAttacks Ver 11 (30-November-2017).csv", 
                              encoding='Windows-1252')




# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities




# convert to lower case
suicide_attacks['City'] = suicide_attacks['City'].str.lower()
# remove trailing white spaces
suicide_attacks['City'] = suicide_attacks['City'].str.strip()




# Your turn! Take a look at all the unique values in the "Province" column. 
# Then convert the column to lowercase and remove any trailing white spaces
Province = suicide_attacks['Province'].unique()
Province.sort()
#Province
suicide_attacks['Province'] = suicide_attacks['Province'].str.lower()
suicide_attacks['Province'] = suicide_attacks['Province'].str.strip()




# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities




# get the top 10 closest matches to "d.i khan"
matches = fuzzywuzzy.process.extract("d.i khan", cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

# take a look at them
matches




# function to replace rows in the provided column of the provided dataframe
# that match the provided string above the provided ratio with the provided string
def replace_matches_in_column(df, column, string_to_match, min_ratio = 90):
    # get a list of unique strings
    strings = df[column].unique()
    
    # get the top 10 closest matches to our input string
    matches = fuzzywuzzy.process.extract(string_to_match, strings, 
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)

    # only get matches with a ratio > 90
    close_matches = [matches[0] for matches in matches if matches[1] >= min_ratio]

    # get the rows of all the close matches in our dataframe
    rows_with_matches = df[column].isin(close_matches)

    # replace all rows with close matches with the input matches 
    df.loc[rows_with_matches, column] = string_to_match
    
    # let us know the function's done
    print("All done!")




# use the function we just wrote to replace close matches to "d.i khan" with "d.i khan"
replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="d.i khan")




# get all the unique values in the 'City' column
cities = suicide_attacks['City'].unique()

# sort them alphabetically and then take a closer look
cities.sort()
cities




# Your turn! It looks like 'kuram agency' and 'kurram agency' should
# be the same city. Correct the dataframe so that they are.
Cities = suicide_attacks['City'].unique()
Cities.sort()
Cities
matches = fuzzywuzzy.process.extract("kurram agency", Cities, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
# only get matches with a ratio > 90
close_matches = [matches[0] for matches in matches if matches[1] >= 96]
# get the rows of all the close matches in our dataframe
rows_with_matches = suicide_attacks['City'].isin(close_matches)

# replace all rows with close matches with the input matches 
suicide_attacks.loc[rows_with_matches, 'City'] = "kurram agency"
Cities = suicide_attacks['City'].unique()
Cities.sort()
Cities
#or just call the function by
#replace_matches_in_column(df=suicide_attacks, column='City', string_to_match="kurram agency", min_ratio=96)




# And that's it for today! If you have any questions, be sure to post them in the comments below or [on the forums](https://www.kaggle.com/questions-and-answers). 

# Remember that your notebook is private by default, and in order to share it with other people or ask for help with it, you'll need to make it public. First, you'll need to save a version of your notebook that shows your current work by hitting the "Commit & Run" button. (Your work is saved automatically, but versioning your work lets you go back and look at what it was like at the point you saved it. It also lets you share a nice compiled notebook instead of just the raw code.) Then, once your notebook is finished running, you can go to the Settings tab in the panel to the left (you may have to expand it by hitting the [<] button next to the "Commit & Run" button) and setting the "Visibility" dropdown to "Public".

# # More practice!
# ___

# Do any other columns in this dataframe have inconsistent data entry? If you can find any, try to tidy them up.

# You can also try reading in the `PakistanSuicideAttacks Ver 6 (10-October-2017).csv` file from this dataset and tidying up any inconsistent columns in that data file.




#baluchistan is misspellt (I checked it in Google). So let's correct it.
# Province
matches = fuzzywuzzy.process.extract("balochistan", Province, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
matches
replace_matches_in_column(df=suicide_attacks, column='Province', string_to_match="balochistan", min_ratio=90)
suicide_attacks['Province'].unique()

