ls ../input

cat ../input/StateNames.csv | head 

import pandas as pd

pres =  {'Name' : ['Barack','George','Bill','George','Ronald','Jimmy','Gerald','Richard','Lyndon','John','Dwight','Harry','Franklin','Herbert','Calvin','Warren','Woodrow'],
         'StartYear' : [2009,2001,1993,1989,1981,1977,1974,1969,1963,1961,1953,1945,1933,1929,1923,1921,1913],
         'EndYear' : [2015,2009,2001,1993,1989,1981,1977,1974,1969,1963,1961,1953,1945,1933,1929,1923,1921]
}

df = pd.DataFrame(pres)

# Find out and Plot names with highest popularity when the current President with same name is in the Office :

import operator
import math
import pandas as pd
import csv

def popularity(selectname):
    #return the year in which name is most popular
    sumname={}
    f = open("../input/StateNames.csv")
    allStates = csv.reader(f)
    for line in allStates:
        state=line[4]
        gender=line[3]
        year=(line[2])
        name=line[1]
        count=(line[5])
        if name==selectname:
            sumname[year]=sumname.get(year,0)+int(count)
    for key,values in sorted(sumname.items(), key=operator.itemgetter(1),reverse=True): #sort by value:
        return key
    f.close()

        
Names = list(df.Name)
print(Names)
startYear=list(df.StartYear)
endYear=list(df.EndYear)
#Names = ['Woodrow','Calvin','Herbert','Franklin','Dwight','Kennedy','Lyndon','Jimmy']
#Names = ['Elvis','Hillary','John','Kennedy']
namesdict ={}
specificName={}
selectedNames =[]

for i in range(0,len(Names)):
    popyear = ''
    popyear = popularity(Names[i])
    if popyear is None:
        continue
    print("The Name " + str(Names[i]) +" is popular during the year " + str(popyear))
    influenceStart = int(startYear[i])-1
    influenceEnd = int(endYear[i])
                           
    if (int(popyear) >= influenceStart and int(popyear) <= influenceEnd): #If popularity is when the president is in office
        selectedNames.append(Names[i])

print("Names with highest popularity when the current President with same name is in the Office :") 
print(selectedNames)

f = open("../input/StateNames.csv")
allStates = csv.reader(f)

for line in allStates:
        state=line[4]
        gender=line[3]
        year=(line[2])
        name=line[1]
        count=(line[5])
        if name in selectedNames:
            specificName = namesdict.get(name,{})
            specificName[year] = specificName.get(year,0)+int(count)
            namesdict[name] = specificName
f.close()


import matplotlib.pyplot as plt
from pylab import rcParams
import operator

#Charts    
rcParams['figure.figsize'] = 17,24
#colors = list("rgbcmykr")
f, axarr = plt.subplots(len(namesdict.items()), sharex=True)
i=0

for name, year_dict in namesdict.items():
    x=[]
    y=[]
    for key,values in sorted(year_dict.items(), key=operator.itemgetter(0)): #sort by year:
        x.append(key) #year
        y.append(values) #counts
    axarr[i].plot(x, y,label=name,linewidth=2,color='b')
    axarr[i].axvspan(int(df[df.Name == name].StartYear), int(df[df.Name == name].EndYear), alpha=0.5, color='red')
    axarr[i].set_title(name)
    i=i+1
print("done")







