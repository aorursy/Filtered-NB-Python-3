#!/usr/bin/env python
# coding: utf-8
Python Premier - Part 1 
1. Assignments and Expressions
2. Conditionals
3. Loops
4. Python Data Structures -  Strings,Lists,Tuples,Sets,Dictionary
5. Comphrensions
6. Functions


# A Sample Program 






# Calculating Simple Interest
principal= 1000 # Initial amount
rate = 0.05      # Interest rate
numyears = 5     # Number of years
year = 1   

while year <= numyears:
    principal = principal*(1+rate)
    print year, principal 
    year += 1




a = 100

1. Assignment operator simply creates association between name and Value
2. Names can refer to different data types during program execution
3. Reassignment binds the name to new value and the old binding is lost 
4. A newline terminates each individual statement.You also can use a semicolon to separate statements. Assignment statement forms
Operation                         Interpretation
spam = 'Spam'                     Basic form
spam, ham = 'yum', 'YUM'          Tuple assignment (positional)
[spam, ham] = ['yum', 'YUM']      List assignment (positional)
a, b, c, d = 'spam'               Sequence assignment, generalized
a, *b = 'spam'                    Extended sequence unpacking (Python 3.X)
spam = ham = 'lunch'              Multiple-target assignment
spams += 42                       Augmented assignment (equivalent to spams = spams + 42)



spam = 'Spam' 
spam




spam, ham = 'yum', 'YUM'
spam,ham




[spam, ham] = ['yum', 'YUM']




spam




ham




a, b, c, d = 'spam' 




a




b




# Calculating Simple Interest
principal= 1000; rate = 0.05; numyears = 5; year = 1;

while year <= numyears:
    principal = principal*(1+rate)
    print year, principal 
    year += 1

1.Format strings contain ordinary text and special formatting-character sequences such as “%d”, “%s”, and “%f”.
2.They specify the formatting of a particular type of data such as an integer, string, or floating-point number.
3.The special character sequences can also contain modifiers that specify a width and precision. 

For example :  “%3d” formats an integer right-aligned in a column of width 3, and “%0.2f” formats a floating-point number so that only two digits appear after the decimal point.


# Calculating Simple Interest
principal= 1000 # Initial amount
rate = 0.05      # Interest rate
numyears = 5     # Number of years
year = 1   

while year <= numyears:
    principal = principal*(1+rate)
    print "%10d %10.3f" % (year,principal) 
    year += 1

   Python Program Heirearchy
1. Programs are composed of modules. 
2. Modules contain statements.
3. Statements contain expressions.
4. Expressions create and process objects.Common Python expression statements

Operation                     Interpretation

spam(eggs, ham)               Function calls
spam.ham(eggs)                Method calls
spam                          Printing variables in the interactive interpreter
print(a, b, c, sep='')        Printing operations in Python 3.X
yield x ** 2                  Yielding expression statements




if test1:          # if test
    statements1    # Associated block
elif test2:        # Optional elifs
    statements2
else:             # Optional else 
    statements3


# Compute the maximum (z) of a and b
if a > b:
    z = a
else:
    z = b




a = 5




b




z




# To create an empty clause, use the pass statement as follows:
if a < b:
    pass       # Do nothing
else:
    z= a




# You can form Boolean expressions by using the or, and, and not keywords:
if b >= a and b <= c:
    print "b is between a and c"
if not (b < a or b > c):
    print "b is still between a and c"




# To handle multiple-test cases, use the elif statement, like this:




if a == '+':
    op = "PLUS"
elif a == '-':
    op = "MINUS"
elif a == '*':
    op = "MULTIPLY"
else:
    raise RuntimeError, "Unknown operator"




print op




# While Loops

while test:        # Loop test
    statements     # Loop body
else:              # Optional else
    statements     # Run if didn't exit loop with break


n = 1000
sum = 0
i = 1




while i <= n:
    sum = sum + i
    i = i + 1




print "Sum of 1 until %d: %d" % (n,sum)




while i <= n:
    sum = sum + i
    i = i + 1
else:
    print "Sum of 1 until %d: %d" % (n,sum)




x = 'spam'
while x:
    print x,
    x = x[1:]




# For Loops

for target in object:   # Assign object items to target 
    statements          # Repeated loop body: use target # Optional else part
else:
    statements         # If we didn't hit a 'break'



sum = 0
for x in [1,2,3,4]:
    sum += x
print sum




T = [(1, 2), (3, 4), (5, 6)]
for (a, b) in T: # Tuple assignment at work
    print(a, b)




D = {'a': 1, 'b': 2, 'c': 3} 
for key in D:             # Use dict keys iterator and index
    print(key, '=>', D[key])




a = range(10)




a 




b = xrange(10)




b









list(b)




a = 12




type(a)




str(a)




print "your age is " + str(a)




float(a)




b = 12.0
type(b)

Octal literals (base 8)
A number prefixed by a 0 (zero) will be interpreted as an octal number


a = 010




a




type(a)

Hexadecimal literals (base 16)
Hexadecimal literals have to be prefixed either by "0x" or "0X".


hex_number = 0xA0F




hex_number




(256* 10) + 15




s = "Hello World"




s




print s




s[0]




s[-1]




s[2:4]




s[2:12:2]




s[-1:-10:-2]




# Operations on  Strings




s




len(s)




name = 'subbu'
print 'Welcome ' + name




s*3




s = 'malayalam'




s == s[-1:-(len(s)+1)]




s == s[-1:-(len(s)+1):-1]




a = [1,2,3,4,5]




a = ['a','b',1,[1,2,3]]




a




len(a)




a[0]




names = ['Dave', 'Mark', 'Ann', 'Phil']




names.append('subbu')




names




names.insert(2,"Ram")




names




names.pop()




names




names.pop(3)




names.remove('Dave')




names




# range 




range(10)




for i in range(10):
    print i**2




# Tuple




names = ('Dave', 'Mark', 'Ann', 'Phil')




len(names)




names.append('Ram')




a,b = 1,2




a




b




a = 7,




a




type(a)




names.append('subbu')




# Sets 




s = set([3,5,9,10])




s




len(s)




type(s)




s.add(11)




s




s.add(11)




s




a = set([1,'a',1.0,(1,2)])




a




t = set('Hello')
print t




t = list('Hello')
print t




s.update(t)




s




# Set Operations




a = set([1,2,3,4,5,6])




b = set([1,2,3,4,5,6,7,8,9,11,22])




a | b      # Union 




a & b     # intersection




b - a   # difference




a ^ b  # not in both




# Dictionary




en_de = {"red" : "rot", "green" : "grun", "blue" : "blau", "yellow":"gelb"}




en_de.keys()




en_de.values()




len(en_de)




del en_de['red']




en_de




'blue' in en_de




for value in en_de.itervalues():
    print value




en_de.items()




# Dictionary from lists




a = [1,2,3]
b = ['one','two','three']




zip(a,b)




dict(zip(a,b))




# Update




w={"house":"Haus","cat":"Katze","red":"rot"}
w1 = {"red":"rouge","blau":"bleu"}




w.update(w1)




w




# Comphrensions




a = range(10)




a




for i in a:
    if i < 6:
        print i**2




# List Comphrensions




[i**2 for i in a if i < 6] 




# Dictionary Comphrensions




num_sq = {i:i**2 for i in range(10)}




num_sq




from math import sqrt




num_sqrt = {k:sqrt(v) for k,v in num_sq.iteritems()}




num_sqrt




# Functions




def welcome(name):
    welcome_msg = "welcome " + name + "!"
    return welcome_msg




welcome("subramanya")




# Default Arguments 




def welcome(name,greet="welcome "):
    welcome_msg = greet + name + "!"
    return welcome_msg




welcome("subramanya")




# Positional arguments




welcome("subramanya","Warm Welcome ")




# Keyword Arguments




welcome(greet = "Warm Welcome ",name="subramanya")




# Argument order in the function




def welcome(greet="welcome ",name):
    welcome_msg = greet + name + "!"
    return welcome_msg




# Lambda




f = lambda word: word.lower()




f("SUBBU")




ops = ['PLUS','MINUS','MULTIPLY']




# map




map(f,ops)




# Filter




filter(lambda op : len(op) < 6,ops)




# reduce

If seq = [ s1, s2, s3, ... , sn ], calling reduce(func, seq) works like this:
    
At first the first two elements of seq will be applied to func, i.e. func(s1,s2) The list on which reduce() works looks now like this: [ func(s1, s2), s3, ... , sn ]
In the next step func will be applied on the previous result and the third element of the list, i.e. func(func(s1, s2),s3)
The list looks like this now: [ func(func(s1, s2),s3), ... , sn ]
Continue like this until just one element is left and return this element as the result of reduce()


reduce(lambda x,y: x+y, [47,11,42,13])




sqrt(4)




import math




math.sqrt(4)




from math import sqrt




sqrt(4)




factorial(5)




from math import *




factorial(5)






