#!/usr/bin/env python
# coding: utf-8



# SETUP. You don't need to worry for now about what this code does or how it works.
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')




def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    return (round(num, 2))
    pass
    
q1.check()




# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()




# Put your test code here
num = round(31.4159, -1)
print(num)

num = round(314.159, -2)
print(num)




#q2.solution()




def to_smash(total_candies, number_of_friends = 3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between numbers_of_friends. If number_of_friends is not
    passed, default value 3 will be used. 
    
    >>> to_smash(91)
    1
    >>> to_smash(91, number_of_friends = 2)
    1
    """
    return total_candies % number_of_friends

q3.check()




#q3.hint()




#q3.solution()




round_to_two_places(9.9999)




x = -10
y = 5
# Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x), abs(y))




def f(x):
    y = abs(x)
    return y

print(f(5))




# Importing the function 'time' from the module of the same name. 
# (We'll discuss imports in more depth later)
from time import time
t = time()
print(t, "seconds since the Epoch")




from time import sleep
duration = 5
print("Getting sleepy. See you in", duration, "seconds")
sleep(duration)
print("I'm back. What did I miss?")




def time_call(fn, arg):
    """Return the amount of time the given function takes (in seconds) when called with the given argument.
    """
    t0 = time()
    fn(arg)
    t1 = time()
    elapsed = t1 - t0
    return elapsed
val = time_call(sleep, 5)
print(val)




#q5.hint()
#q5.solution()




def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    return max(time_call(fn, arg1), time_call(fn, arg2), time_call(fn, arg3))
    




#q6.hint()




#q6.solution()

