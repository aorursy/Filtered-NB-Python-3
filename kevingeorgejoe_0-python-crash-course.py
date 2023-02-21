#!/usr/bin/env python
# coding: utf-8



a = 2
print(a, type(a))




a = 4.2
print(a, type(a))




a = "hey there"
print(type(a), a)

a = "hi"
print(type(a), a)

a = 'i'
print(type(a), a)

a = "d"
print(type(a), a)




b = 1

sum = a + b
print(sum, type(sum))




print(z)




z = "I'm down here. Help :("




a = "hey" + 'there'
print(a, type(a))




str1 = "Papa Kevin"
str2 = "Mama Kevin"
str3 = "Kevin 11"
str4 = "7-11"

print(str1.endswith("Kevin"))
print(str2.endswith("in"))
print(str2.endswith("  11"))




type(str1.endswith("Kevin"))




bool




list1 = list()
list2 = []

print("list1:", list1)
print("list2:", list2)




list1.append("a")
list1.append("b")
list1.append("c")

# printing entire list
print(list1)

# printing the zeroth element of the list
print(list1[0])

# printing the second element of the list
print(list1[2])




print(list1[3])




cholesterol = "hi"
# cholesterol = "med"
# cholesterol = "low"
# cholesterol = "over level 9000"

if (cholesterol == "hi"):
    print("Get help, now!")
elif (cholesterol == "med"):
    print("Take it easy!")
else:
    print("Enjoy :)")




movieRating = 3
# movieRating = 9.5
# movieRating = 7
# movieRating = 5.4
# movieRating = 9999
# movieRating = -2

if (movieRating > 8 and movieRating >=10):
    print("Loved")
elif (movieRating > 6):
    print("Liked it")
elif (movieRating > 4):
    print("Ok")
elif (movieRating > 2):
    print("Disliked it")
elif (movieRating > 0):
    print("Hated it")
else:
    print("Invalid movieRating")
    




sunOut = True
clouds = False
rain = False
snow = False

if (sunOut):
    print("Sunny")
elif (clouds):
    if (rain):
        print("Rainy")
        print("Take an Umbrella!")
    else:
        print("Cloudy")
elif (snow):
    print("Snowing")
    print("Wear a jacket")
else:
    print("It's cloudy with a chance of meatballs")




sunOut = False
clouds = True
rain = True
snow = False

if (sunOut):
    print("Sunny")
elif (clouds):
    if (not rain):
        print("Cloudy")
    else:
        print("Rainy")
        print("Take an Umbrella!")        
elif (snow):
    print("Snowing")
    print("Wear a jacket")
else:
    print("It's cloudy with a chance of meatballs")




sunOut = False
clouds = False
rain = False
snow = False

if (sunOut):
    print("Sunny")
elif (clouds):
    if (not rain):
        print("Cloudy")
    else:
        print("Rainy")
        print("Take an Umbrella!")        
elif (snow):
    print("Snowing")
    print("Wear a jacket")
else:
    print("It's cloudy with a chance of meatballs!")




def predictWeather(sunOut, clouds, rain, snow):
    if (sunOut):
        print("Sunny")
    elif (clouds):
        if (not rain):
            print("Cloudy")
        else:
            print("Rainy")
            print("Take an Umbrella!")        
    elif (snow):
        print("Snowing")
        print("Wear a jacket")
    else:
        print("It's cloudy with a chance of meatballs!")
    




sunOut = True
clouds = False
rain = False
snow = False

predictWeather(sunOut, clouds, rain, snow)




predictWeather(True, False, False, False)




predictWeather(sunOut=True, clouds=False, rain=False, snow=False)




predictWeather(clouds=False, rain=False, snow=False, sunOut=True)




def predictWeather(sunOut, clouds, rain, snow=False):
    if (sunOut):
        print("Sunny")
    elif (clouds):
        if (not rain):
            print("Cloudy")
        else:
            print("Rainy")
            print("Take an Umbrella!")        
    elif (snow):
        print("Snowing")
        print("Wear a jacket")
    else:
        print("It's cloudy with a chance of meatballs!")




predictWeather(False, False, False)




def predictWeather(sunOut, clouds, rain, snow=True):
    if (sunOut):
        print("Sunny")
    elif (clouds):
        if (not rain):
            print("Cloudy")
        else:
            print("Rainy")
            print("Take an Umbrella!")        
    elif (snow):
        print("Snowing")
        print("Wear a jacket")
    else:
        print("It's cloudy with a chance of meatballs!")




predictWeather(False, False, False)




def predictWeather(sunOut, clouds=True, rain, snow=True):
    if (sunOut):
        print("Sunny")
    elif (clouds):
        if (not rain):
            print("Cloudy")
        else:
            print("Rainy")
            print("Take an Umbrella!")        
    elif (snow):
        print("Snowing")
        print("Wear a jacket")
    else:
        print("It's cloudy with a chance of meatballs!")




print("ACM")
print("ACM")
print("ACM")
print("ACM")
print("ACM")
print("ACM")
print("ACM")




for i in range(0, 10, 1):
    print(i)




for i in range(0, 10, 3):
    print(i)




for i in range(0, 10):
    print(i)




print(list1)




for element in list1:
    print(element)




import math

a = 2
result = math.pow(a, 5)
print(result)




print(a ** 5)




a = 5
math.sin(a)




math.pi




list2 = [1, math.pi, 89, -3]

for element in list2:
    math.sin(element)
    
print(list2)




list2 = [1, math.pi, 89, -3]

for elem in list2:
    elem = math.sin(elem)
    
print(list2)




for i, element in enumerate(list2):
    list2[i] = math.sin(element)
    
print(list2)

