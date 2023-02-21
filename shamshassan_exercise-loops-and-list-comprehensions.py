#!/usr/bin/env python
# coding: utf-8



from learntools.core import binder; binder.bind(globals())
from learntools.python.ex5 import *
print('Setup complete.')




def has_lucky_number(nums):
    nums=[28]
    for num in nums:
        if num % 7 == 0:
            return True
        else:
            return False




def has_lucky_number(nums):
    if not nums: 
        return False
    else:
        for num in nums:
            if num % 7 == 0:
                return True
    return False
    
       

q1.check()
numbers=[28,14,35,20]
has_lucky_number(numbers) 




#q1.hint()
#q1.solution()




[1, 2, 3, 4] > 2




def elementwise_greater_than(L, thresh):
   
   # for i in L:
   #     if L[i] > thresh:
   #         print('True')
   #     else:
   #         print('False')
    status = []
    for i, num in enumerate(L):
        status.append(True if num>thresh else False)
    return status

s=[1,2,3,4]
number=2
elementwise_greater_than(s,number)

q2.check()




q2.solution()




def menu_is_boring(meals):
     for i in range(len(meals)):
        if(i<len(meals)-1):
            if(meals[i+1]==meals[i]): 
                return True
    return False
m=['Spam', 'Eggs', 'Spam', 'Spam', 'Bacon', 'Spam']
menu_is_boring(m)



q3.check()




#q3.hint()
#q3.solution()




play_slot_machine()




def estimate_average_slot_payout(n_runs):
    i =0
    while i<n_runs:
        profit = play_slot_machine() - 1
        i+= 1
    return profit/n_runs
n_runs=200
estimate_average_slot_payout(n_runs)




q4.solution()

