#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




#We import some library numpy and pandas for now and take our data from csv file 
data = pd.read_csv('../input/countries of the world.csv')




#Info gives general information of csv file; for instance null or non-null, object(string), float,integer,range index,data columns,memory usage and category name
data.info()




data.corr()




#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()




#columns give all category name like country, region, population etc.
data.columns




#head gives the information that you want number of countries
data.head(5)




#it gives count, mean, standard derivative,min,max and other information 
data.describe()




'''Variables'''
#There are mainly five type of variable.These are integer, float,boolean,complex and string

number=9 #integer variable
number2=7.5 #float variable
boolean=True#boolean variable
complex1=4+3j#complex variable
string="Arif Mandal" #string variable

print(number)
print(number2)
print(boolean)
print(complex1)
print(string)

'''Maths Operations'''
#Summation
print(number+number2)
#Substraction
print(number-number2)
#multiplication
print(number*number2)
#division
print(number/number2)
#exact division
print(number//number2)
#module
print(number%number2)

#Type function
print(type(number))
print(type(number2))
print(type(boolean))
print(type(complex1))
print(type(string))




#Actually converting each other

number=5.4
print(number)
print(int(number))#it gives exact value 5

string="45"
print(string)
print(type(string))
newVar=int(string)
print(newVar)
print(type(newVar))




string="Arif Mandal" #Index of string start 0,1,2,3...
print(string[1])#it gives indexing character

print(string[0:3])# start from 0(0 include) but 3 not include

print(string[:])#all string

print(string[::-1])#reverse of string

print(len(string))#it gives length of string

print(string.lower())#returns the string in lower case

print(string.upper())#returns the string in upper case

print(string.split(","))#returns ["Arif","mandal"]

print(string.replace("A","D"))#replaces a string with another string

print(string.isdigit())#it checks digit or not so return false

print(string.isalpha())#it checks alpha or not so return false

print(string.isprintable())#it checks printable or not so return true

print(string.index("a"))#returns index of a


#Escape Sequence
'''
\newline	Ignored
\\	Backslash (\)
\'	Single quote (')
\"	Double quote (")
\a	ASCII Bell (BEL)
\b	ASCII Backspace (BS)
\f	ASCII Formfeed (FF)
\n	ASCII Linefeed (LF)
\r	ASCII Carriage Return (CR)
\t	ASCII Horizontal Tab (TAB)
\v	ASCII Vertical Tab (VT)

'''

#you can find all function with below command
#print(dir(string))#in here string is type of python

#this is not important (for now) just it gives useful function.While you working use this function methods
for x in dir(string):
    if "_" not in x:
        print(x)





list=["arif","mandal",4,7,5]#create list

print(list)#returns all elements

print(list[0])#returns 0 index of list

list[3]=45#list is changeable
print(list)

print(len(list))#returns length of list


list.append("value")#append methods provide to add new element
print(list)

list.pop()#pop methods provide to remove last element
print(list)

list.remove("arif")#remove methods provide to remove element
print(list)


list.clear()#delete all elements in list
print(list)


#the other methods
for x in dir(list):
    if "_" not in x:
        print(x)
        




tuple1=("arif","mandal",1,2)#create a tuple

print(tuple1)

print(tuple1[1])#access tuple items by referring to the index number

#tuple1[1]="value"
#print(tuple1)#Tuples are unchangeable.

print(len(tuple1))#returns length of list

#the other methods:
for x in dir(tuple):
    if "_" not in x:
        print(x)





set1={"python","java","c","c++"}#create a set

#Note:Sets are unordered, so the items will appear in a random order.
print(set1)

#write all elements in set1
for i in set1:
    print(i)
    
    
set1.add("php")#add an item to a set
print(set1)


set1.remove("java")#remove an item to a set,but If the item to remove does not exist, it raise error
#set1.discard("java")#the same with remove
print(set1)

print(len(set1))#returns length of set1


set1.clear()#returns empty set
print(set1)

#the other set
set2={"Turkey","USA","England"}
set3={"Turkey","Poland","England"}

set2.pop()#delete one item in set2
print(set2)

print(set2.intersection(set3))#returns common elements in set2 and set3

#the other methods:
for x in dir(set):
    if "_" not in x:
        print(x)




dict1={"C":"Ken Thompson","C++":"Bjarne Stroustrup","Java":"James Gosling","Python":" Guido van Rossum"}

# Dictionaries are unordered, changeable and indexed and are written with curly brackets, and they have keys and values.
print(dict1)

print(dict1["Java"])#Access the items of a dictionary by referring to its key name

for i in dict1:#Access all keys with for loop
    print(i)
    
for i,j in dict1.items():#Access all keys and values with for loop
    print(i,"---",j)


    
dict1["C"]="Arif Mandal:)"#Change the value of a specific item by referring to its key name
print(dict1)


print(len(dict1))#returns length of dict1


del dict1["Java"]#delete a spesific item
print(dict1)

dict1.clear() #delete all items
print(dict1)


#the other methods:
for x in dir(dict):
    if "_" not in x:
        print(x)


#Actually some methods are similar with list, tuple and set, so it is normal to forget,,,don't worry just look at these methods with using. 





#we use it to get input from the user.

user_input=input("What is your name: ")
print("Welcome ",user_input)

#input fuction accept only string variable

number=input("Enter the any number: ")
number2=5
#print(number+number2)

#gave a this error; TypeError: must be str, not int 
#thus first must convert int


number=int(input("Enter the any number: "))
number2=5
print(number+number2)#now there is not any problem


#the other solution
number=int(input("Enter the any number: "))
number2=5
print(int(number)+number2)#now there is not any problem






#These are usual logical conditions:
#1)Equals: a == b
#2)Not Equals: a != b
#3)Less than: a < b
#4)Less than or equal to: a <= b
#5)Greater than: a > b
#6)Greater than or equal to: a >= b

num1=20
num2=30

#version1
if num2>num1: #if statement to test whether num2 is greater than num1. if num2 greater than num1; print the screen else nothing
    print("Number2 greater than number1")



#version2
if num2>num1:
    print("Number2 greater than number1")
elif num2<num1:#if the previous conditions were not true, then try this condition
    print("Number1 greater than number2")
elif num2==num1:
    print("Number1 equals number2")
    


#version3
if num2>num1:
    print("Number2 greater than number1")
elif num2<num1:
    print("Number1 greater than number2")
elif num2==num1:
    print("Number1 equals number2")
else:#catches anything which isn't caught by the preceding conditions.
    print("Some error but I dont know...")



#Short hand
print("num1") if num1 > num2 else print("=") if num1 == num2 else print("num2")


b1=False
b2=True

#and
if b1 and b2:#checks two variable value if someone is false returns false
    print("False")
else:
    print("True")
    

#or
if b1 or b2:##checks two variable value if someone is true returns true
    print("False")
else:
    print("True")

    
#EXAMPLE:
#1)odd or even:

user_input=int(input("Enter a number: "))

if user_input%2==0:
    print("Even")
else:
    print("Odd")

#2)Age Limit:
user_input=int(input("Enter your age: "))

if user_input>18:
    print("You are adult")
elif user_input<18 and user_input>10:
    print("You are teenager")
else:
    print("You are baby")




#It has the ability to iterate over the items of any sequence, such as a list or a string.
count=1
for i in "Python":#Print each letter in a Python:
    print(str(count)+".Letter is "+i)
    count+=1
    
    
print("******************************************************************")
list1=["Java","Python","C","C++","Php","Arduino"]

for i in list1:#Print each language in a list1:
    print(i)

    

print("******************************************************************")

for i in list1:#we can stop the loop using if conditionals
    if i is "C":#is equals ==
        break
    print(i)
    


print("******************************************************************")


for i in list1:#we can stop the current iteration of the loop, and continue with the next
    if i is "C":#is equals ==
        continue
        
    print(i)   

    
print("******************************************************************")

for i in range(0,10,1):#range fuction take 3 variable first=start ,second=stop(not include), third=increment rate
    print(i)


print("******************************************************************")

dict1={"1":1,"2":2,"3":3,"4":4,"5":5}

for i,j in dict1.items():#we use for loop in dict to write keys and values 
    print(i,"equals",str(j))
    

print("******************************************************************")
#the multiplication table using for loop
for i in range(1,11):
    for j in range(1,11):
        print("ixj",(i*j))
    print("")
        
    
#Example1):odd or even

for i in range(0,11):
    if i%2==0:
        print(str(i),"Even")
    else:
         print(str(i),"Odd")
            
print("******************************************************************")            
#Example2): Prime numbers
print(2) 
for i in range(3,20):

    div = False
    for j in range(2,i):
            if i % j == 0:
                div=True
                
    if div == False:
        print (i)
           
      




#while loop; we can execute a set of statements as long as a condition is true.

num = 1
while num < 10:#Print i as long as i is less than 10
    print(num)
    num+=1
    
print("******************************************************************")
    

num2=5
while num2<9:#like for loop
    if num2==7:
        break
    print(num2)
    num2+=1

    
print("******************************************************************")


num3=5
while num3<9:#like for loop
    num3+=1
    if num3==7:
        continue
    print(num3)
    
    
#while True:#infinite loop
    #print("infinite")
    
    
    
#general I prefer for loop




#A function is a block of code which write one use all time.It is very usefull tools in programming.


def write_Name():#In Python a function is defined using the def keyword
    print("Arif Mandal")
    print("Python")

#calling function as below   
write_Name()
write_Name()
write_Name()
write_Name()
print("******************************************************************")
def summation(num1,num2):#Information can be passed to functions as parameter.num1,num2 are parameter and you write many parameters as you want, just separate them with a comma
    print(num1+num2)
    

summation(45,25)
summation(20,36)
print("******************************************************************")

#default parameter
def country(country="Turkey"):
    print("My country ", country)


country()
country("USA")

print("******************************************************************")

#we use return in function
def evenOdd(num):
    if num%2==0:
        return "Even"
    else:
        return "Odd"


evenOdd(12)

print("******************************************************************")

#flexible parameters

def mult(*args):
    result=1
    for i in args:
        result *= i
    return result
    
print(mult(1,1,2,3,5,8,13,21,34))


#Example1)Fibonacci 

def fib(num):
    if num==1:
        return [1]
    elif num==2:
        return [1, 1]
    else:
        lst = fib(num - 1)
        return lst + [lst[-1] + lst[-2]] 
    
fib(10)




#A lambda function can take any number of arguments, but can only have one expression.

#lambda function seems function but it is easy and fast.
mult=lambda x,y: x*y

print(mult(15,10))

print("******************************************************************")
#factorial calculator with using lambda 
fact = lambda x: 1 if x == 0 else x * fact(x-1)

print(fact(5))

print("******************************************************************")

#square calculation of number with using lambda 
square=lambda x: x*x

print(square(7))




#General format of try except
try:
    print(x)
except:
    print("Error...") #because x not defined
    
print("******************************************************************")
#Actually this error type NameError. Let's try this.

try:
    print(y)
except NameError:
    print("Value error your programming includes not defined variable.")
    
print("******************************************************************")
#We can define as many exception blocks as you want
try:
    print(var)
except NameError:
    print("Name Error.")
    
except:
    print("Something is wrong. ")
    
print("******************************************************************")
#finally block:
#it will be executed regardless if the try block raises an error or not.

try:
    x=15/0
    print(x)
except ZeroDivisionError:
    print("Zero division error.")
    
finally:
    print("Codes continues in here...")
    
    
print("******************************************************************")


#Using Raise:
raise NameError('HiThere')
    
    
print("******************************************************************")
#Some Error Example
#1)Index error:
try:
    list1=["python","java",1889,1994]
    print(list1[4])
except IndexError:
    print("INDEX ERROR...")

    
print("******************************************************************")
#2)IOError:

try: 
    fileName="file.txt"
    f=open(fileName,"r")
except IOError:
    print("IOError... File not exits")
    
print("******************************************************************")
#3)ValueError:

while True:
    try:
        x = int(input("Please enter a number: "))
        break
    except ValueError:
        print("That was no valid number.Try again...")
 




import math as m

'''pi'''

print(m.pi)#returns pi number

'''e'''

print(m.e)#returns e number that natural logarithm

'''tau'''

print(m.tau)#returns ratio of a circle’s circumference to its radius. 

'''inf'''

print(m.inf)#returns positive infinity

'''nan'''
print(m.nan)#returns “not a number” (NaN) value

'''Ceil()'''

x=5.6

print(m.ceil(x)) #Returns the smallest integer greater than or equal to x

'''floor(x)'''

x=4.5

print(m.ceil(x))#Returns the largest integer less than or equal to x

'''fabs()'''

x=-4

print(m.fabs(x))#Returns the absolute value of x

'''factorial(x)'''

print(m.factorial(4))#returns factorial of number

'''fmod(x, y)'''

print(m.fmod(47,2))#Returns the remainder when x is divided by y

'''fsum(iterable)'''

l=[4,5,7,8]

print(m.fsum(l))#Returns an accurate floating point sum of values in the iterable

'''isfinite(x)'''

x=float('inf')

print(m.isfinite(x))#Returns True if x is neither an infinity nor a NaN (Not a Number)

'''exp(x)'''

print(m.exp(4))#Returns e**x

'''expm1(x)'''

print(m.expm1(4))#Returns e**x - 1

'''log(x[, base])'''

print(m.log(10))#Returns the logarithm of x to the base (defaults to e)

'''log2(x)'''

print(m.log2(10))#Returns the base-2 logarithm of x

'''pow(x, y)'''

print(m.pow(4, 4))#Returns x raised to the power y

'''sqrt(x)'''

print(m.sqrt(81))#Returns the square root of x

'''asin(x)'''

print(m.asin(m.sqrt(3)/2))#Returns the arc sine of x
#acos(x),atan(x) use same

'''sin(x)'''

print(m.sin(m.pi/2))#Returns the sine of x
#cos(x),tan(x) use same

'''asinh(x)'''

print(m.asinh(1/2))#Returns the inverse hyperbolic sine of x
#acosh(x),atanh(x) use same

'''sinh(x)'''

print(m.sinh(m.pi/2))#Returns the hyperbolic cosine of x
#cosh(x),tanh(x) use same





import random as r

'''random()'''

print(r.random())# returns between 0 and 1, including 0; but not including 1 [0,1)

'''randint(a, b)'''

print(r.randint(4,10))#returns integer value between [4,10] this mean include 4 and 10

'''randrange(start, stop[, step])'''

print(r.randrange(0,8,2))#Returns a number that is not included in the max in the min and max range
#first parameter is start point, second parameter is stop point, last parameter is step amount

'''sample(list,q)'''

numbers=range(50)

print(r.sample(numbers,7))#In the list, q returns the random value.

'''Choice'''

l=[1,2,3,4,5,6,7,8,9,10,11,12]

x=r.choice(l)

print(x)#choice one eleman that created list and returns

'''Shuffle'''

l=[4,5,6,7,8,9,10,11,12,12,14,15]

x=r.shuffle(l)

print(l)#changes the order of elements in the created list

'''uniform(a, b)'''

print(r.uniform(0, 9))#Return a random floating point number between a and b inclusive





import time as t

print(t.time())#returns from January 1, 1970, how many seconds has passed.

print(t.localtime())#returns local time

print(t.asctime())#retuns day,month,time and year

print(t.strftime('%c'))#the other methods to take date and time 

'''
        *Date Formers*
'''

print(t.strftime("%Y"))

print(t.strftime("%B"))

print(t.strftime("%A"))

print(t.strftime("%d/%m/%Y"))#returns formated date

t.sleep(2)#This code will stop our program for 2 seconds.

for i in range(0,5): #Using this function, we can interrupt the operation of our codes for certain periods of time.
    t.sleep(3)
    print("Arif Mandal")
    




'''abs()'''

x=-20

print(abs(x))#abs stand for absolute so returns positive value of variable

'''round()'''

print(round(22/7))#exact result must be 3.14 round function roll closest number.Also,
#the function take 2 parameter first parameter original number, second parameter how many digit that you want

print(round(22/7, 3))

'''all()'''

l=[1,2,4,7,8,]

print(all(l))#returns true if all values in a list are True.If any of these values is False, then False is to return the value. 

l2=[0,2,4,6]

print(all(l2))

'''any()'''

l3=["Arif","Mandal",""]

print(any(l3))#The task of any () function If at least one of all values in an array is True, True is output.

'''bin()'''

print(bin(12))#returns binary format

'''ord()'''

print(ord("a"))#Returns the decimal number that a character corresponds to.

print(ord("q"))

'''oct()'''

print(oct(24))#returns a number to its equivalent in octal order

'''hex()'''

print(hex(32))#returns a number to its equivalent in hexadecimal order

'''copyright()'''

print(copyright())#You may access Python's copyright information.

'''dir()'''

print(dir())#If we use dir () without parameters, we get a list of items in the current namespace

'''divmod()'''

print(divmod(10,5))#The first parameter returns the partition portion of the partition operation and the second element returns the remainder.

'''enumerate()'''

print(list(enumerate('Python')))#returns numbered list.

for i in enumerate('Python'):#the other way
     print(i)

'''exit()'''

#exit()  #This function allows you to exit the currently running program.

'''help()'''

print(help(dir))#Using this function, we can access the help documentation for items in the Python programming language.

'''filter()'''

print([i for i in l if i % 2 == 1 ])#this is classic way of find odd numbers

def odd(number):
    return number%2==1

print (*filter(odd, l))#With this built-in function, we can apply a filtering based on a specific criterion on the elements in array objects.

'''len()'''

string="Python"

l=["python","java","php","javascript","c"]

print(len(string))#returns length of string, list, set etc.

print(len(l))

'''map()'''

l=[1,2,3,4,5,6,7,8,9]

def square(l):
    return l**2

print(list(map(square,l)))#takes 2 parameter first parameter is intended function, other parameter is list

'''max()'''

tuples=(1,4,7,25,69,12)

print(max(tuples))#returns maximum value in tuples, list etc.

languages =[ 'python' , 'ruby' , 'go' , 'r' , 'java' , 'assembly' ]

print(max(languages, key=len ))#string version

'''mix()'''

tuples=(1,4,7,25,69,12)

print(min(tuples))#it is similar with max it returns minimum value in tuples, list etc

languages =[ 'python' , 'ruby' , 'go' , 'r' , 'java' , 'assembly' ]

print(min(languages, key=len ))#string version

'''pow'''

print(pow(3,2))#returns 3^2=9

print(pow(3,2,3))#The third parameter allows us to calculate the modulus of the number obtained by the force calculation.

'''quit()'''

#quit() #we use to exit the kernel

'''range()'''

for i in range(0,10,2):#we use to list numbers in a certain range.
    print(i)
    

'''reversed()'''

languages =[ 'python' , 'ruby' , 'go' , 'r' , 'java' , 'assembly' ]

print(list(reversed(languages)))#returns inverse list

print(languages[::-1])#classic way that learned before

'''sorted()'''

languages =[ 'python' , 'ruby' , 'go' , 'r' , 'java' , 'assembly' ]

numList=[4,45,12,5,89,24]

print(list(sorted(languages)))#returns sorted list.

print(list(sorted(numList)))#for numbers

'''sum()'''

numList=[4,45,12,5,89,24]

print(sum(numList))#returns summation of numList

'''type()'''

print(type("Turkey"))#it is to say what type of data an object belongs to.

'''zip()'''

a1=["a","b","c","d","e"]

a2=[1,2,3,4,5]

print(list(zip(a1,a2)))#returns matching items by order

print(*zip(a1,a2))#second way

for i,j in zip(a1,a2):#third way
    print(i,j)

    
'''vars()'''

print(vars(str))#returns methods and properties of objects

'''
End of Built-in Function 
Thank you
'''




'''Introduction Of Class'''

class Student:#class name; Generally, first letter becomes capital letter
    
    country="Turkey" # class attribute
    
    def __init__(self, name,surname,age,school):# instance attribute
        self.name = name
        self.surname=surname
        self.age = age
        self.school=school
        
# instantiate the Student class        
st1=Student("Arif","Mandal",21,"AGU")
st2=Student("Ceyhun","Buyuk",20,"ODTU")

print(st1.country)

print("St1 name {} , surname {} , age {} , school {}".format(st1.name,st1.surname,st1.age,st1.school))#show the screen

print("St2 name {} , surname {} , age {} , school {}".format(st2.name,st2.surname,st2.age,st2.school))#show the screen







        
    
        




'''Methods in Class'''
class Student:#class name; Generally, first letter becomes capital letter
    
    country="Turkey" # class attribute
    
    def __init__(self, name,surname,age,school):# instance attribute
        self.name = name
        self.surname=surname
        self.age = age
        self.school=school
    
    #Setter and Getter
    #NAME
    def setName(self,name):
        self.name=name
    def getName(self):
        return self.name
    
    #SURNAME
    def setSurname(self,surname):
        self.surname=surname
    def getSurname(self):
        return self.surname
    
    #AGE
    def setAge(self,age):
        self.age=age
    def getAge(self):
        return self.age
    
    #SCHOOL
    def setSchool(self,school):
        self.school=school
    def getSchool(self):
        return self.school
    
    #Other Method
    
    #SHOW information
    
    def showInfo(self):
        print("All Information:")
        print('''                 Name= {}
                 Surname={}
                 Age={}
                 School={}
                 '''.format(self.name,self.surname,self.age,self.school))
                
        
# instantiate the Student class        
st1=Student("Arif","Mandal",21,"AGU")
st2=Student("Ceyhun","Buyuk",20,"ODTU")

print("St1 name {} , surname {} , age {} , school {}".format(st1.name,st1.surname,st1.age,st1.school))

st1.setName("Kerim")

print("St1 name {} , surname {} , age {} , school {}".format(st1.name,st1.surname,st1.age,st1.school))

n=st1.getName()
print(n)

print("Student1")
st1.showInfo()

print("Student2")
st2.showInfo()




'''Example of Class(Circle)'''

class Circle:
    pi=3.14
    
    # instance attribute
    def __init__(self,radius=2):#radius=2 default value; if user not defined radius, radius it will be 2
        self.radius=radius
    
    #setter and getter
    
    def setRadius(self,radius):
        self.radius=radius
    
    def getRadius(self):
        return self.radius
    
    #AREA METHODS
    def area(self):
        return (self.radius**2)*Circle.pi
    
    #CIRCUMFERENCE METHODS
    def cir(self):
        return 2*Circle.pi*self.radius
    


#create object

c1=Circle()

print("Default radius of circle 1 is ", c1.getRadius())#default value of Circle

c1.setRadius(4)

area=c1.area()

print("Area of circle 1=",area)

circumference=c1.cir()

print("Circumference of circle 1=",circumference)

#create other object
c2=Circle(10)

print("c2 radius=",c2.getRadius())

print("Area of circle 2=",c2.area())

print("Circumference of circle 2=",c2.cir())

        
    




'''Example of Class(Point)'''

class Point:
    
    def __init__(self,x1,y1,x2,y2):
        self.x1=x1
        self.y1=y1
        self.x2=x2
        self.y2=y2
    
    #GETTER AND SETTER
    
    def setx1(self,x1):
        self.x1=x1
    
    def getx1(self):
        return x1

    def sety1(self,y1):
        self.y1=y1
    
    def gety1(self):
        return y1
    
    def setx2(self,x2):
        self.x2=x2
    
    def getx2(self):
        return x2

    def sety2(self,y2):
        self.y2=y2
    
    def gety2(self):
        return y2
    
    #distance
    
    def distance(self):
        return ((self.x1-self.x2)**2 +(self.y1-self.y2)**2)**0.5
    
    #slope
    
    def slope(self):
        return int((self.y1-self.y2)/(self.x1-self.x2))
    

p1=Point(1,2,3,4)

print("A(",p1.x1,",",p1.y1,")","B(",p1.x2,",",p1.y2,")")

print("Distance two point=",p1.distance())

print("Slope of two point=",p1.slope())

p1.setx1(5)
p1.sety2(8)

print("New slope of two point=",p1.slope())
    




'''Inheritance'''

# parent class
class Animal:
    
    def __init__(self):
        print("Animal created")
        
    def who(self):
        print("Animal")
        
    def eat(self):
        print("Eating...")
    
    def sleep(self):
        print("Sleeping...")
        

# child class

class Bird(Animal):
    
    def __init__(self):
        # call super() function
        print("Bird created")
        super().__init__()
        
    def who(self):
        print("Bird")     
        
    def fly(self):
        print("Fyling...")
        

flappy=Bird()

print(flappy.who())

print(flappy.eat())

print(flappy.sleep())

print(flappy.fly())


        









import numpy as np #import numpy library as np

'''Intro'''

array=np.array([1,2,3,4])#create numpy array

print(array)

print(type(array))

print(array.shape)#returns type of matris that mean how many rows and how many columns  

print(array[0])#Access any eleman using list method in numpy array

array[2]=12 #Change an element of the array

print(array)#print new array on the screen

print("******************************************************************")

'''Some important methods that give information of array '''

array2=np.array([[12,23,14],[25,36,78]]) #create 2 dimension array

print(array2)

print(array2.shape) 

print(array2.ndim)#returns dimension of array

print(array2.size)#returns size of array how many number or eleman include in array

print(array2.dtype)# returns data type of array.Additionally NumPy provides types of its own. numpy.int32, numpy.int16, and numpy.float64 are some examples.

print(array2.itemsize)#the size in bytes of each element of the array. 

print("******************************************************************")

'''Create some types arrays in Numpy'''

print(np.zeros(5))  # Create an array of all zeros

print(np.zeros((2,3)))#2 dimension zeros array

print(np.ones((4,4))) # Create an array of all ones

print(np.full((3,3), 3)) # Create a constant array

print(np.eye(2))#identity array

print(np.empty([4, 2],dtype=int)) #we decide data type of array also empty function create random number in matrix

print(np.arange(3,7)) #returns an array that between start point to stop point.

print(np.linspace(5.0,7.0, num=5))#returns an array that split n point between start point to stop point

print(np.random.rand(2,2))

print("******************************************************************")

'''Basic Operations'''
a=np.array([1,4,7])
b=np.array([2,5,8])

summation=a+b
subtraction=a-b

mult=a*b # elementwise product
mult1=a@b # matrix product
mult2=a.dot(b) # another matrix product

mult3=a*5 
mult4=10*np.sin(b) 

div=a/b
square=np.sqrt(a)

boolean=a>1

x=a.sum()#returns summation of each element
y=a.min()#returns minimum element of a matris
z=a.max()#returns maximum element of a matris


print(summation)
print(subtraction)
print(mult)
print(mult1)
print(mult2)
print(mult3)
print(mult4)
print(div)
print(square)
print(boolean)
print(x)
print(y)
print(z)

print("******************************************************************")

newArray=np.array([[12,24,25],[21,47,65]])

print(np.sum(newArray, axis=0)) # Compute sum of each column
print(np.sum(newArray,axis=1)) # Compute sum of each row

print("******************************************************************")

'''Universal Functions'''

array3=np.array([1,2,3,4,5,6,7,8,9])

print(np.mean(array3))#returns the average of the array elements.

print(np.median(array3))#Returns the median of the array elements.

print(np.std(array3))#Returns the standard deviation, a measure of the spread of a distribution, of the array elements.****std = sqrt(mean(abs(x - x.mean())**2)).***

print(np.sort(array3))#returns sorted matris

print(np.transpose(array3))#returns tranpose of matrix

print(np.var(array3))#Returns the variance of the array elements, a measure of the spread of a distribution. ***var = mean(abs(x - x.mean())**2).***

print(np.cumsum(array3))#Return the cumulative sum of the elements along a given axis.

print("******************************************************************")

'''Indexing, Slicing'''

array4=np.array([0,1,8,27,64,125,216,343,512,729])

print(array4[5])

print(array4[3:7])

print(array4[: : -1])  # reversed array4

#for loop

for i in array4:
    print(i**2)#returns square of each eleman
    
    
array5 = np.array([[1, 2], [3, 4]])

for index, x in np.ndenumerate(array5):
     print(index, x)
    

        
print("******************************************************************") 

'''Shape Manipulation'''

array6=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(array6.shape)

print(array6.ravel())#returns flattened array

print(array6.reshape(6,2))#returns 6x2 matris 

#Several arrays can be stacked together along different axes
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])

print(np.hstack((a,b)))#x axis

print(np.vstack((a,b)))#y axis

print("******************************************************************") 

'''Copies'''

c=a.copy()

print(c)

c[0,1]=148

print(a[0,1])

print(c[0,1])

print("******************************************************************") 

'''End of Numpy Thank you'''
'''Next Topic is Pandas...'''




import pandas as pd #import pandas library

world=pd.read_csv('../input/countries of the world.csv')#Let's read our csv file and assign it to a variable.

world #returns all table

world.head(10) #returns the first five rows and it depends on parameter in head

world.tail(10)##returns the last five rows and it depends on parameter in tail

world.shape #returns number of rows and columns 

world["Population"] #returns information of population

world.Population #the other way

world.index#returns start point to ending point and step number

world.loc[0:10,:]#display the ten rows of world using the loc method

world.loc[:10,"Region"]#returns special columns

world.loc[:10,["Region","Population"]]#returns special columns

world.info()#returns all information of data like below
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 227 entries, 0 to 226
Data columns (total 20 columns):
Country                               227 non-null object
Region                                227 non-null object
Population                            227 non-null int64
Area (sq. mi.)                        227 non-null int64
Pop. Density (per sq. mi.)            227 non-null object
Coastline (coast/area ratio)          227 non-null object
Net migration                         224 non-null object
Infant mortality (per 1000 births)    224 non-null object
GDP ($ per capita)                    226 non-null float64
Literacy (%)                          209 non-null object
Phones (per 1000)                     223 non-null object
Arable (%)                            225 non-null object
Crops (%)                             225 non-null object
Other (%)                             225 non-null object
Climate                               205 non-null object
Birthrate                             224 non-null object
Deathrate                             223 non-null object
Agriculture                           212 non-null object
Industry                              211 non-null object
Service                               212 non-null object
dtypes: float64(1), int64(2), object(17)
memory usage: 35.5+ KB
'''


'''Creating A DataFrame in Pandas'''

data=pd.DataFrame(world)#Creating A DataFrame in Pandas

type(data)#retuns pandas.core.frame.DataFrame

data.describe() #returns statistics value like count, mean, std,min,max etc.

data["Population"]#returns population's information like above I telling.

'''Adding Columns'''

from random import randint

random_numbers = []
for i in range(0,227):
    random_numbers.append(randint(1,228))
    
    
data['id'] = random_numbers #new column name is id

data.head()

'''Deleting Colums'''

data.drop('id',axis=1, inplace=True)

data.head()


'''Sorting'''

data.sort_values("Country",ascending=False)#sorted table

data.head()

data.sort_values(["Country","Region"])

data.head()


'''Filtering'''

density = []
for i in data.Population:
    if i <= 60000000:
        density.append('Low')
    elif i > 60000000 and i <=120000000:
        density.append('Medium')
    else:
        density.append('High')

data['Density'] = density

data.head(10)

data[data.Density == 'High']

data[data.Density == 'Medium']

data[data.Density == 'Low']

data[(data.Region == 'WESTERN EUROPE') | (data.Density == 'Medium')]# or 

data[(data.Region == 'WESTERN EUROPE') & (data.Density == 'High')]# and


'''Some Important Methods'''
#It is very usefull methods be careful...

data.columns # returns category name 

data.corr #returns all table

data['Population'].describe() #returns statistic information of data

#Also we take one by one

data['Population'].mean()

data['Population'].min()

data['Population'].max()

data['Population'].std()

data['Population'].mode()

data.isnull().any().any() #it returns True or False depend on, if data includes null value, returns True, otherwise False

data=data.dropna() #convert all null value to 0

data.isnull().any().any() #Now, it retuns False

'''End of Pandas Thank you'''
'''Next Topic is Seaborn...'''




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

world=pd.read_csv('../input/countries of the world.csv')

data=pd.DataFrame(world)

world.fillna(0.0, inplace = True)



def convertValue(val):
    newVal = val.replace(',','.')
    return float(newVal)


'''Bar Plot '''
    
    
data["Region"] = data["Region"].astype("category")
data["Country"] = data["Country"].astype('category')

data["Region"] = data["Region"].str.strip()

group = data.groupby("Region")
group.mean()

data.info()

region = data["Region"].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=region.index,y=region.values)
plt.xticks(rotation=90)
plt.ylabel('Number of countries')
plt.xlabel('Region')
plt.title('Number of Countries by REGİON',color = 'red',fontsize=20)
plt.plot()




'''Other Bar Plot'''

climate = data["Climate"].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=climate.index,y=climate.values)
plt.xticks(rotation=90)
plt.ylabel('Number of climate')
plt.xlabel('Degree of climate')
plt.title('Number of climate by degree',color = 'blue',fontsize=20)

plt.plot()




'''Box Plot'''
group = data.groupby("Region")
group.mean()

sns.boxplot(x=data["Region"],y=data["GDP ($ per capita)"],data=data, width=0.7,palette="Set2",fliersize=5)
plt.xticks(rotation=90)
plt.title("GDP BY REGİON",color="green")

plt.plot()




'''Other Box Plot'''

group = data.groupby("Region")
group.mean()

sns.boxplot(x=data["Region"],y=data["Population"],data=data, width=0.7,palette="Set2",fliersize=5)
plt.xticks(rotation=90)
plt.title("Population BY REGİON",color="red")

plt.plot()




data.corr()

f,ax = plt.subplots(figsize=(18, 16))
sns.heatmap(data.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)

plt.plot()




'''Joint Plot'''

data.info()

data["Deathrate"] = data["Deathrate"].astype(str)
data["Deathrate"] = data["Deathrate"].apply(convertValue)
data["GDP ($ per capita)"] = data["GDP ($ per capita)"].astype(str)
data["GDP ($ per capita)"] = data["GDP ($ per capita)"].apply(convertValue)

g = sns.jointplot(data["Deathrate"], data["GDP ($ per capita)"], kind="kde", size=7)
plt.savefig('graph.png')
plt.show()




'''Other Joint Plot'''

data["Birthrate"] = data["Birthrate"].astype(str)
data["Birthrate"] = data["Birthrate"].apply(convertValue)
data["GDP ($ per capita)"] = data["GDP ($ per capita)"].astype(str)
data["GDP ($ per capita)"] = data["GDP ($ per capita)"].apply(convertValue)

g = sns.jointplot(data["Birthrate"], data["GDP ($ per capita)"], kind="kde", size=7)
plt.savefig('graph.png')
plt.show()




'''DistPlot'''

sns.set(style="darkgrid", palette="muted",font_scale=1)
sns.distplot(data["Birthrate"],bins=20,kde=True,color="y")
plt.tight_layout()#Increases the alignment of the drawn graph.
plt.plot()




sns.set(style="white",palette="Blues", font_scale=1)
sns.distplot(data["GDP ($ per capita)"],hist=False,bins=20,kde=True,color="g",kde_kws={"shade":True})
plt.tight_layout()#Increases the alignment of the drawn graph.
plt.plot()




#we can show all graphics with subplots 

sns.set(style="darkgrid", palette="muted",font_scale=1)

f,axes=plt.subplots(2,2,figsize=(15,10))

sns.distplot(data["Birthrate"],bins=20,kde=True,color="y",ax=axes[0,0])

sns.distplot(data["GDP ($ per capita)"],hist=False,bins=20,kde=True,color="g",kde_kws={"shade":True},ax=axes[0,1])

sns.distplot(data["Population"],hist=True,bins=20,kde=True,color="g",kde_kws={"shade":True},ax=axes[1,0])

sns.distplot(data["Deathrate"],bins=20,kde=False,color="y",ax=axes[1,1])

plt.plot()




'''Joint Plot Continue'''

sns.jointplot(data["Deathrate"],data["Birthrate"],data=data)
plt.plot()





data["Industry"] = data["Industry"].astype(str)
data["Industry"] = data["Industry"].apply(convertValue)
data["Service"] = data["Service"].astype(str)
data["Service"] = data["Service"].apply(convertValue)

sns.jointplot(data["Industry"],data["Service"],data=data,color="r",kind="hex")

plt.plot()





sns.jointplot(data["Industry"],data["Service"],data=data,color="r",kind="reg")

plt.plot()




sns.jointplot(data["Industry"],data["Service"],data=data,color="r",kind="kde")

plt.plot()




'''Pair Plot'''

sns.pairplot(data,palette="inferno", hue="Region",)

plt.plot()






'''Violin Plot'''

sns.set(style="whitegrid")
sns.violinplot(x="Industry",y="Service",data=data,hue="Region",palette="PRGn")

plt.plot()






'''Strip Plot'''

sns.stripplot(x="Region",y="Population",data=data,color="m")

plt.xticks(rotation=90)

plt.plot()







'''Factor Plot'''

sns.factorplot(x="Region",y="Population",data=data,color="m", kind="point",palette="muted",size=8)

plt.xticks(rotation=90)

plt.plot()




'''LmPlot'''#it is important for machine learning

sns.lmplot(x="Birthrate",y="Deathrate",data=data,size=10,hue="Region")

plt.plot()






#we can use to see detailed

sns.lmplot(x="Birthrate",y="Deathrate",data=data,col="Region",size=10)

plt.plot()




'''Scatter'''

data.info()

plt.scatter(data["Birthrate"],data["Deathrate"],marker='^',facecolor='green')

plt.grid(True)

plt.xlabel('Birthrate')
plt.ylabel('Deathrate')

plt.title("Scatter Plot")

plt.legend(loc='upper left')

plt.show()




'''Histogram'''

data.info()

plt.hist(data["Service"], bins=10,density=True, facecolor='g', alpha=0.75)

plt.xlabel('Service')

plt.ylabel('Rate')

plt.title('Histogram Plot')

plt.grid(True)

plt.show()




'''Pie Chart'''

explode = (0, 0.1, 0, 0,0,0,0)

sizes=[15,10,25,5,30,5,10]

labels="ASIA","EASTERN EUROPE","NORTHERN AFRICA","OCEANIA","WESTERN EUROPE","SUB-SAHARAN AFRICA","NORTHERN AMERICA"

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels,explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()







'''Line Plot'''

plt.plot(data["Birthrate"])

#plt.xlim(5, 0)

plt.xlabel('Rate')

plt.ylabel('Birthrate')

plt.title('Line Plot')

plt.grid(True)

plt.show()









**If you like please votes up...If you have any questions, comment...Thank you**

