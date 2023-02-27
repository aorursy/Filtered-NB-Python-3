#!/usr/bin/env python
# coding: utf-8



from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import *
print('Setup complete.')




# Your code goes here. Define a function called 'sign'
def sign(no_to_check):
    if no_to_check < 0:
        return -1
    elif no_to_check > 0:
        return 1
    else:
        return 0

q1.check()




#q1.solution()




def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    print("Splitting", total_candies, "candies")
    return total_candies % 3

to_smash(81)




to_smash(1)




def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    if total_candies == 1:
        print("Splitting", total_candies, "candy")
    else:
        print("Splitting", total_candies, "candies")
    return total_candies % 3

to_smash(91)
to_smash(1)




#q2.solution()




## 3. <span title="A bit spicy" style="color: darkgreen ">🌶️</span>

# In the main lesson we talked about deciding whether we're prepared for the weather. I said that I'm safe from today's weather if...
# - I have an umbrella...
# - or if the rain isn't too heavy and I have a hood...
# - otherwise, I'm still fine unless it's raining *and* it's a workday

# get_ipython().set_next_input('The function below uses our first attempt at turning this logic into a Python expression. I claimed that there was a bug in that code. Can you find it');get_ipython().run_line_magic('pinfo', 'it')

# To prove that `prepared_for_weather` is buggy, come up with a set of inputs where it returns the wrong answer.




def prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday):
    # Don't change this code. Our goal is just to find the bug, not fix it!
    return (have_umbrella or rain_level < 5) and (have_hood or not (  rain_level > 0 and is_workday))

# Change the values of these inputs so they represent a case where prepared_for_weather
# returns the wrong answer.
have_umbrella = False
rain_level = 0.0
have_hood = False
is_workday = False

# Check what the function returns given the current values of the variables above
actual = prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday)
print(actual)

q3.check()




q3.hint()
q3.solution()




#def is_negative(number):
  # """ if number < 0:
  #      return True
   # else:
    #    return False"""

def concise_is_negative(number):
    return  number < 0
print(concise_is_negative(9))
q4.check()




q4.hint()
q4.solution()




def onionless(ketchup, mustard, onion):
    """Return whether the customer doesn't want onions.
    """
    return not onion




def wants_all_toppings(ketchup, mustard, onion):
    """Return whether the customer wants "the works" (all 3 toppings)
    """
    #if ketchup and mustard and onion:
     #   return True
   # else:
    #    return False
    return ketchup and mustard and onion



q5.a.check()




q5.a.hint()
q5.a.solution()




def wants_plain_hotdog(ketchup, mustard, onion):
    """Return whether the customer wants a plain hot dog with no toppings.
    """
    #if not ketchup and not mustard and not onion:
     #   return True
    #else:
     #   return False
    return not (ketchup or mustard or onion)

q5.b.check()




#q5.b.hint()
#q5.b.solution()




def exactly_one_sauce(ketchup, mustard, onion):
    """Return whether the customer wants either ketchup or mustard, but not both.
    (You may be familiar with this operation under the name "exclusive or")
    """
    return (ketchup and not mustard) or (mustard and not ketchup)

        #return True
   # else:
       # return False

q5.c.check()




q5.c.hint()
q5.c.solution()




def exactly_one_topping(ketchup, mustard, onion):
    """Return whether the customer wants exactly one of the three available toppings
    on their hot dog.
    """
    return (ketchup + mustard + onion) == 1

q6.check()




q6.hint()
q6.solution()




def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay.
    When calculating a hand's total value, we count aces as "high" (with value 11) if doing so
    doesn't bring the total above 21, otherwise we count them as low (with value 1). 
    For example, if the player's hand is {A, A, A, 7}, we will count it as 11 + 1 + 1 + 7,
    and therefore set player_total=20, player_low_aces=2, player_high_aces=1.
    """
    return False




q7.simulate_one_game()




q7.simulate(n_games=50000)




def should_hit(dealer_total, player_total, player_low_aces, player_high_aces):
    """Return True if the player should hit (request another card) given the current game
    state, or False if the player should stay.
    When calculating a hand's total value, we count aces as "high" (with value 11) if doing so
    doesn't bring the total above 21, otherwise we count them as low (with value 1). 
    For example, if the player's hand is {A, A, A, 7}, we will count it as 11 + 1 + 1 + 7,
    and therefore set player_total=20, player_low_aces=2, player_high_aces=1.
    """
    return False

q7.simulate(n_games=50000)

