#!/usr/bin/env python
# coding: utf-8



# SETUP. You don't need to worry for now about what this code does or how it works. If you're ever curious about the 
# code behind these exercises, it's available under an open source license here: https://github.com/Kaggle/learntools/
import sys; sys.path.insert(0, '../input/learntools/pseudo_learntools')
from learntools.python import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')




def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    # Replace this body with your own code.
    # ("pass" is a keyword that does literally nothing. We used it as a placeholder
    # because after we begin a code block, Python requires at least one line of code)
    #pass
    num_rounded=round(num,2)
    return num_rounded

q1.check()




# Uncomment the following for a hint
#q1.hint()
# Or uncomment the following to peek at the solution
#q1.solution()




numbr=10.56785
num_rounded=round(numbr,-1)
print(num_rounded)
#q2.solution()




def to_smash(total_candies,num_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    
    return total_candies % num_friends

q3.check()




#q3.hint()




#q3.solution()




round_to_two_places(9.9999)




x = -10
y = 5
# # Which of the two variables above has the smallest absolute value?
smallest_abs = min(abs(x),abs(y))
print(smallest_abs)




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
    #pass
    t1=time()
    fn(arg)
    t2=time()
    t_diff=t2-t1
    print(t_diff, "seconds taken for the function")
    return t_diff

time_call(f,5)




#q5.hint()




#q5.solution()




def slowest_call(fn, arg1, arg2, arg3):
    """Return the amount of time taken by the slowest of the following function
    calls: fn(arg1), fn(arg2), fn(arg3)
    """
    #arg 1 case
#     t1=time()
#     fn(arg1)
#     t2=time()
    t_diff1=time_call(fn, arg1)
    
    #arg 2 case
#     t3=time()
#     fn(arg2)
#     t4=time()
    t_diff2=time_call(fn, arg2)
    
    #arg 3 case
#     t5=time()
#     fn(arg3)
#     t6=time()
    t_diff3=time_call(fn, arg3)
    
    t_diff_min=min(t_diff1,t_diff2,t_diff3)
    
    print(t_diff1, "seconds taken for the function for arg1")
    print(t_diff2, "seconds taken for the function for arg2")
    print(t_diff3, "seconds taken for the function for arg3")
    
    return t_diff_min

print("The slowest call duration is:",slowest_call(f,2, 2.0, 2.00))
    #pass




#q6.hint()




#q6.solution()




print(print("Spam"))




q7.hint()




# Uncomment for an explanation.
q7.solution()




def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    return min(s1, s2, s3)

smallest_stringy_number('10', '3', '2')




q8.a.solution()




def smallest_stringy_number(s1, s2, s3):
    """Return whichever of the three string arguments represents the smallest number.
    
    >>> smallest_stringy_number('1', '2', '3')
    '1'
    """
    
    return min(s1,s2,s3,key=int)

q8.b.check()




q8.b.hint()




#q8.b.solution()

