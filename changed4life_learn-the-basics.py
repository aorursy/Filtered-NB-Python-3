#!/usr/bin/env python
# coding: utf-8



print("This line will be printed.")




print("Hello World")




x = 1
if x == 1:
    # Indented four spaces
    print("x is 1.")




# Modify below to print "Hello, World"
print("Goodbye, World!")




# Comments are added to code to explain or add content to your code. They are ignored by the complier.
print ('Hello World!') # You can even add them to the end of a line. The code before the comment will still be executed.

"""
You
can
even
do
multi-line
docstrings
"""
# These are compiled but are just printed out as strings to the console




myint = 7
print(myint)




myfloat = 7.0
print(myfloat)
myfloat = float(7)
print(myfloat)




mystring = 'hello'
print(mystring)
mystring = "hello"
print(mystring)




mystring = "Don't worry about apostrophes"
print(mystring)




one = 1
two = 2
three = one + two
print(three)

hello = "hello"
world = "world"
helloworld = hello + " " + world
print(helloworld)




a, b = 3, 4
print(a, b)




# This will not work!
one = 1
two = 2
hello = "hello"

print(one + two + hello)




# Change this code
mystring = None
myfloat = None
myint = None

# Testing code
if mystring == "hello":
    print("String: %s" % mystring)
if isinstance(myfloat, float) and myfloat == 10.0:
    print("Float: %f" % myfloat)
if isinstance(myint, int) and myint == 20:
    print("Integer: %d" % myint)




mylist = []
mylist.append(1)
mylist.append(2)
mylist.append(3)
print(mylist[0]) # prints 1
print(mylist[1]) # prints 2
print(mylist[2]) # prints 3

# Prints out 1, 2, 3
for x in mylist:
    print(x)




mylist = [1, 2, 3]
print(mylist[10])




list_len = len(mylist)
print(list_len)




list_min = min(mylist)
list_max = max(mylist)
print(list_min)
print(list_max)




mylist.insert(3, 2.5)
print(mylist)




mylist.pop(3)
print(mylist)




mylist.remove(2)
print(mylist)




mylist.extend([6, 7, 8])
print(mylist)




list_sum = sum(mylist)
print(list_sum)




print(mylist)
mylist.sort(reverse=True)
print(mylist)
mylist.sort(reverse=False)
print(mylist)




numbers = []
strings = []
names = ["John", "Eric", "Jessica"]

# Write your code here
second_name = None


# This code should write out the filled arrays and the second name in the names list (Eric).
print(numbers)
print(strings)
print("The second name on the names list is %s" % second_name)




number = 1 + 2 * 3 / 4.0
print(number)




int_div = 5 // 3
real_div = 5.0 / 3
print(int_div)
print(real_div)




remainder = 11 % 3
print(remainder)




squared = 7 ** 2
cubed = 2 ** 3
print(squared)
print(cubed)




helloworld = "hello" + " " + "world"
print(helloworld)




lotsofhellos = "hello" * 10
print(lotsofhellos)




even_numbers = [2, 4, 6, 8]
odd_numbers = [1, 3, 5, 7]
all_numbers = odd_numbers + even_numbers
print(all_numbers)




print([1, 2, 3] * 3)




x = object()
y = object()

# TODO: change this code
x_list = [x]
y_list = [y]
big_list = []

print("x_list contains %d objects" % len(x_list))
print("y_list contains %d objects" % len(y_list))
print("big_list contains %d objects" % len(big_list))

# Testing code
if x_list.count(x) == 10 and y_list.count(y) == 10:
    print("Almost there...")
if big_list.count(x) == 10 and big_list.count(y) == 10:
    print("Great!")




# This prints out "Hello, John!"
name = "John"
print("Hello, %s!" % name)




# This prints out "John is 23 years old."
name = "John"
age = 23
print("%s is %d years old." % (name, age))
print("%s is %03d years old." % (name, age))
print("%s is %3d years old." % (name, age))
print("%s is %2.2f years old." % (name, age))




# This prints out: A list: [1, 2, 3]
mylist = [1, 2, 3]
print("A list: %s" % mylist)




# This prints out "Hello, John!"
name = "John"
print("Hello, {}!".format(name))




# This prints out "John is 23 years old."
name = "John"
age = 23
print("{0} is {1:0=2d} years old.".format(name, age))
print("{0} is {1:0=3d} years old.".format(name, age))
print("{0} is {1:=3d} years old.".format(name, age))
print("{0} is {1:=2.2f} years old.".format(name, age))




# This prints out: A list: [1, 2, 3]
mylist = [1, 2, 3]
print("A list: {}".format(mylist))




data = ("John", "Doe", 53.44)
format_string = "Hello"

print(format_string % data)




astring = "Hello world!"
astring2 = 'Hello world!'




astring = "Hello world!"
print("single quotes are ' '")

print(len(astring))




astring = "Hello world!"
print(astring.index("o"))




astring = "Hello world!"
print(astring.count("l"))




astring = "Hello world!"
print(astring[3:7])




astring = "Hello world!"
print(astring[3:7:2])




astring = "Hello world!"
print(astring[3:7])
print(astring[3:7:1])




astring = "Hello world!"
print(astring[::-1])




astring = "Hello world!"
print(astring.upper())
print(astring.lower())




astring = "Hello world!"
print(astring.startswith("Hello"))
print(astring.endswith("asdfasdfasdf"))




astring = "Hello world!"
afewwords = astring.split(" ")




s = "Hey there! what should this string be?"
# Length should be 20
print("Length of s = %d" % len(s))

# First occurrence of "a" should be at index 8
print("The first occurrence of the letter a = %d" % s.index("a"))

# Number of a's should be 2
print("a occurs %d times" % s.count("a"))

# Slicing the string into bits
print("The first five characters are '%s'" % s[:5]) # Start to 5
print("The next five characters are '%s'" % s[5:10]) # 5 to 10
print("The thirteenth character is '%s'" % s[12]) # Just number 12
print("The characters with odd index are '%s'" % s[1::2]) #(0-based indexing)
print("The last five characters are '%s'" % s[-5:]) # 5th-from-last to end

# Convert everything to uppercase
print("String in uppercase: %s" % s.upper())

# Convert everything to lowercase
print("String in lowercase: %s" % s.lower())

# Check how a string starts
if s.startswith("Str"):
    print("String starts with 'Str'. Good!")

# Check how a string ends
if s.endswith("ome!"):
    print("String ends with 'ome!'. Good!")

# Split the string into three separate strings, each containing only a word
print("Split the words of the string: %s" % s.split(" "))

# Reverse the string
s_list = list(s)
s_reversed = ''.join(s_list[::-1])
print("Reversed string: %s" % s_reversed)




x = 2
print(x == 2) # prints out True
print(x == 3) # prints out False
print(x < 3) # prints out True




name = "John"
age = 23
if name == "John" and age == 23:
    print("Your name is John, and you are also 23 years old.")

if name == "John" or name == "Rick":
    print("Your name is either John or Rick.")
    
if type(name) == str:
    print ("It is a string!")
    
if isinstance(age,int):
    print ("It is an int!")




name = "John"
if name in ["John", "Rick"]:
    print("Your name is either John or Rick.")




statement = False
another_statement = True
if statement is True:
    # Do something
    pass
elif another_statement is True: # else if
    # Do something else
    pass
else:
    # Do another thing
    pass




x = 2
if x == 2:
    print("x equals two!")
else:
    print("x does not equal to two.")




x = [1, 2, 3]
y = [1, 2, 3]
print(x == y) # prints out True
print(x is y) # prints out False




print(not False) # prints out True
print((not False) == (False)) # prints out False




# Change this code
number = 10
second_number = 10
first_array = []
second_array = [1, 2, 3]

if number > 15:
    print("1")

if first_array:
    print("2")

if len(second_array) == 2:
    print("3")

if len(first_array) + len(second_array) == 5:
    print("4")

if first_array and first_array[0] == 1:
    print("5")

if not second_number:
    print("6")




primes = [2, 3, 5, 7]
for prime in primes:
    print(prime)




# Prints out the numbers 0, 1, 2, 3, 4
for x in range(5):
    print(x)

# Prints out 3, 4, 5
for x in range(3, 6):
    print(x)

# Prints out 3, 5, 7
for x in range(3, 8, 2):
    print(x)




# Prints out 0, 1, 2, 3, 4

count = 0
while count < 5:
    print(count)
    count += 1  # this is the same as count = count + 1




# Prints out 0, 1, 2, 3, 4

count = 0
while True:
    print(count)
    count += 1
    if count >= 5:
        break

# Prints out only odd numbers - 1, 3, 5, 7, 9
for x in range(10):
    # Check if x is even
    if x % 2 == 0:
        continue
    print(x)




# Prints out 0, 1, 2, 3, 4 and then it prints "count value reached 5"

count = 0
while(count < 5):
    print(count)
    count +=1
else:
    print("count value reached %d" % count)

# Prints out 1, 2, 3, 4
for i in range(1, 10):
    if(i % 5 == 0):
        break
    print(i)
else:
    print("this is not printed because for loop is terminated because of break but not due to fail in condition")




numbers = [
    951, 402, 984, 651, 360, 69, 408, 319, 601, 485, 980, 507, 725, 547, 544,
    615, 83, 165, 141, 501, 263, 617, 865, 575, 219, 390, 984, 592, 236, 105, 942, 941,
    386, 462, 47, 418, 907, 344, 236, 375, 823, 566, 597, 978, 328, 615, 953, 345,
    399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217,
    815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 742, 717,
    958, 609, 842, 451, 688, 753, 854, 685, 93, 857, 440, 380, 126, 721, 328, 753, 470,
    743, 527
]

# Your code goes here




# block_head:
#     1st block line
#     2nd block line
#     ...




def my_function():
    print("Hello From My Function!")




def my_function_with_args(username, greeting):
    print("Hello, %s , From My Function!, I wish you %s" % (username, greeting))




def sum_two_numbers(a, b):
    return a + b




# Define our 3 functions
def my_function():
    print("Hello From My Function!")

def my_function_with_args(username, greeting):
    print("Hello, %s, From My Function! I wish you %s" % (username, greeting))

def sum_two_numbers(a, b):
    return a + b

# Prints a simple greeting
my_function()

# Prints "Hello, John Doe, From My Function!, I wish you a great year!"
my_function_with_args("John Doe", "a great year!")

# After this line x will hold the value 3!
x = sum_two_numbers(1, 2)




# Modify this function to return a list of strings as defined above
def list_benefits():
    pass

# Modify this function to concatenate to each benefit - " is a benefit of functions!"
def build_sentence(benefit):
    pass

def name_the_benefits_of_functions():
    list_of_benefits = list_benefits()
    for benefit in list_of_benefits:
        print(build_sentence(benefit))

name_the_benefits_of_functions()




class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")




class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()




class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

myobjectx.variable




class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

print(myobjectx.variable)




class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()
myobjecty = MyClass()

myobjecty.variable = "yackity"

# Then print out both values
print(myobjectx.variable)
print(myobjecty.variable)




class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass()

myobjectx.function()




# Define the Vehicle class
class Vehicle:
    name = ""
    kind = "car"
    color = ""
    value = 100.00
    def description(self):
        desc_str = "%s is a %s %s worth $%.2f." % (self.name, self.color, self.kind, self.value)
        return desc_str
# Your code goes here

# Test code
print(car1.description())
print(car2.description())




phonebook = {}
phonebook["John"] = 938477566
phonebook["Jack"] = 938377264
phonebook["Jill"] = 947662781
print(phonebook)




phonebook = {
    "John" : 938477566,
    "Jack" : 938377264,
    "Jill" : 947662781
}
print(phonebook)




phonebook = {"John" : 938477566,"Jack" : 938377264,"Jill" : 947662781}
for name, number in phonebook.items():
    print("Phone number of %s is %d" % (name, number))




phonebook = {
   "John" : 938477566,
   "Jack" : 938377264,
   "Jill" : 947662781
}
del phonebook["John"]
print(phonebook)




phonebook = {
   "John" : 938477566,
   "Jack" : 938377264,
   "Jill" : 947662781
}
phonebook.pop("John")
print(phonebook)




phonebook = {
    "John" : 938477566,
    "Jack" : 938377264,
    "Jill" : 947662781
}

# Write your code here


# Testing code
if "Jake" in phonebook:
    print("Jake is listed in the phonebook.")
if "Jill" not in phonebook:
    print("Jill is not listed in the phonebook.")




mygame/
mygame/game.py
mygame/draw.py




# game.py

# Import the draw module
import draw

def play_game():
    ...

def main():
    result = play_game()
    draw.draw_game(result)

# This means that if this script is executed, then main() will be executed
if __name__ == '__main__':
    main()




# draw.py

def draw_game():
    ...

def clear_screen(screen):
    ...




# game.py

# Import the draw module
from draw import draw_game

def main():
    result = play_game()
    draw_game(result)




# game.py

# Import the draw module
from draw import *

def main():
    result = play_game()
    draw_game(result)




# game.py

# Import the draw module
if visual_mode:
    # In visual mode, we draw using graphics
    import draw_visual as draw
else:
    # In textual mode, we print out text
    import draw_textual as draw

def main():
    result = play_game()
    # This can either be visual or textual depending on visual_mode
    draw.draw_game(result)





import os
os.environ['PYTHONPATH']='/foo'
import foo




import sys
sys.path.append('/foo')




# Import the library
import urllib

# Use it
x = urllib.request.urlopen('https://www.google.com/')
print(x.read())




import urllib
dir(urllib)




help(urllib.error)




import foo.bar




from foo import bar




import re

# Your code goes here

