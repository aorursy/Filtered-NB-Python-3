import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr  
import os

result = pd.read_csv('../input/primary_results.csv')
result = result.sort_values(['fips', 'candidate'], ascending=[1, 1])
result[:10]

# Add a column with the total votes each candidate got
aggregated = result.groupby('candidate').sum()['votes']
aggregated.name = 'sum_votes'
aggregated[:10]
result1 = result.join(aggregated,on='candidate')
result1[:20]

# Add a column with the total votes each candidate got
aggregated = result.groupby('candidate').sum()['votes']
aggregated.name = 'sum_votes'
aggregated[:10]
result1 = result.join(aggregated,on='candidate')
result1[:20]

#Create a pair-wise list of candidates
unique_pair = []
pairs = [[i,j] for i in ['Donald Trump', 'Ben Carson', 'John Kasich', 'Ted Cruz', 'Marco Rubio']
               for j in ['Donald Trump', 'Ben Carson', 'John Kasich', 'Ted Cruz', 'Marco Rubio'] if i != j]

for pair in pairs:
    if list(reversed(pair)) in unique_pair:
        continue
    unique_pair.append(pair)
print (unique_pair)

#Calculate the Pearson correlation
cor = []
for pair in unique_pair:
    ['{} vs. {}'.format(i[0], i[1]), 
                (pearsonr(Rep_res_name[pair[0], Rep_res_name[pair[1]]**2]
    cor.append()

max = ['',.5]
min = ['',.5]
for i in corrVals:
    if max[1] < i[1]:
        max = i
    if min[1] > i[1]:
        min = i

print('Max correlation: {}\nMin correlation: {}'.format(max, min))
