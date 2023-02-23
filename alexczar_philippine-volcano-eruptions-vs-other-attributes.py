#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from scipy import stats

# data processing
data = pd.read_csv('volcano.csv', encoding='latin1')




E = data['Eruptions']




# Eruptions vs latitude
C = data['Coordinates']

# Strip latitudes
C = [int(l[:2]) for l in C]

print("Eruptions vs latitude (N): ", stats.pearsonr(C, E)[0])




# Output
# Eruptions vs latitude (N):  0.0375643739884
    
# (a little correlation -> the higher north the volcano, the more eruptions it has)




# Eruptions vs longitude
C = data['Coordinates']

# Strip longitudes
C = [int(l.split(' ')[1][:3]) for l in C]

print("Eruptions vs longitude (W): ", stats.pearsonr(C, E)[0])




# Output
# Eruptions vs longitude (W):  -0.0721502591359

# (a little correlation -> the further east the volcano, the less eruptions it has)




# Eruptions vs volcano height in meters
C = data['m']

print("Eruptions vs height (m): ", stats.pearsonr(C, E)[0])




# Output
# Eruptions vs height (m):  0.242221970196

# (a little correlation -> the taller the volcano, the more eruptions it has)

