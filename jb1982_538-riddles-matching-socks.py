#!/usr/bin/env python
# coding: utf-8



# import modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random




def messy_sock_drawer(no_of_pairs = 10, sims = 1):
   
   # initialise a list to store results of simulations
   pulls = []
   
   for sim in range(sims):

       # initialise variables and empty lists
       drawer = []
       removed_socks = []

       # create a drawer full of socks, and shuffle them about
       for sock in range(1, no_of_pairs + 1):
           drawer += [sock] * 2
       random.shuffle(drawer)

       # continue to pull socks out of the drawer, until you have two of a kind
       while True:
           pulled_sock = drawer[0]
           removed_socks.append(drawer[0])
           del drawer[0]
           if removed_socks.count(pulled_sock) > 1:
               break

       # calculate the number of pulls it took to get two of a kind
       pulls_needed = len(removed_socks)
       pulls.append(pulls_needed)
   
   return pulls




# run the function for ten pairs of socks, and a 10k simulations
pulls = messy_sock_drawer(10, 10000)
average_pulls =np.mean(pulls)
print("The average number of pulls to return a pair is: " + str(average_pulls))




# plot the results of the above simulation
sns.set_style('ticks')
sns.countplot(pulls, color = 'lightsteelblue')
sns.despine(top = True, right = True)
plt.xlabel("pulls needed")
plt.ylabel("count")
plt.title("Pulls needed to return a pair (100k simulations)")
plt.show()

