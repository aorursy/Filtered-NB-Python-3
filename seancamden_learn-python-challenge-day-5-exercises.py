#!/usr/bin/env python
# coding: utf-8



# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys
sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex5 import *
print('Setup complete.')




nums = [1,2,3,5]
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    if nums:
        luck = False
        for num in nums:
            if num % 7 == 0:
                luck = True
        if luck == True:
            return True
        else:
            return False
    else:
        return False
        
has_lucky_number(nums)




def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    for num in nums:
        if num % 7 == 0:
            return True
    return False
q1.check()




q1.hint()
#q1.solution()




# [1, 2, 3, 4] > 2




# def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is 
    True if L[i] is greater than thresh, and False otherwise.
    
    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """
    # return [t > thresh for t in L]

# q2.check()




#q2.solution()




# def menu_is_boring(meals):
#     """Given a list of meals served over some period of time, return True if the
#     same meal has ever been served two days in a row, and False otherwise.
#     """
#     print(meals)
#     if len(meals) > 1:
#         for i in meals:
#             m1 = meals[-1]
#             m2 = meals[-2]
#             print(m1,m2,len(meals),sep="/")
#             if m1 == m2:
#                 return True
#             else:
#                 meals.pop()
#         return False
#     else:
#         return False
# q3.check()




# q3.hint()
# #q3.solution()




# play_slot_machine()




# def estimate_average_slot_payout(n_runs):
#     """Run the slot machine n_runs times and return the average net profit per run.
#     Example calls (note that return value is nondeterministic!):
#     >>> estimate_average_slot_payout(1)
#     -1
#     >>> estimate_average_slot_payout(1)
#     0.5
#     """
#     runs = [play_slot_machine() for i in range(n_runs)]
#     avg = (sum(runs) - n_runs) / n_runs

#     return avg

# # account for $1 per run
# estimate_average_slot_payout(1000000)




# q4.solution()




# def slots_survival_probability(start_balance, n_spins, n_simulations):
#     """Return the approximate probability (as a number between 0 and 1) that we can complete the 
#     given number of spins of the slot machine before running out of money, assuming we start 
#     with the given balance. Estimate the probability by running the scenario the specified number of times.
    
#     >>> slots_survival_probability(10.00, 10, 1000)
#     1.0
#     >>> slots_survival_probability(1.00, 2, 1000)
#     .25
#     favorable outcomes versus total outcomes
#     start_balance + winnings = total_spins
#     """
#     if start_balance >= n_spins:
#         p = 1.0
#         return p
#     elif start_balance < n_spins:

# #q5.check()
# slots_survival_probability(10,12,1000000)




# q5.solution()

