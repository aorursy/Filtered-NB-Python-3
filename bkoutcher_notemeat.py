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




# Returns the matches that have valid players.
# Guarantees that there are AT LEAST the specified number of
# valid players (num_players) for both the Home and Away team.
find_non_null_matches <- function(matches, num_players) {
if (num_players <= 11 && num_players > 0) {
non_null_matches = matches
for (i in c(1:num_players)) {
non_null_matches = non_null_matches[which((non_null_matches[,(55+i)] != "NA")
& (non_null_matches[,(66+i)] != "NA")),]
}
return (non_null_matches)
} else {
return (matches)
}
}

