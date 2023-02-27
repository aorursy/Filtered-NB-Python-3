
import pandas as pd

NationalNames = pd.read_csv('../input/NationalNames.csv')
NationalNames[NationalNames['Name'] == 'Lyndon'][NationalNames['Gender'] == 'M'].plot(x='Year',y='Count',figsize=(17,4))




pres =  {'Name' : ['Barack','George','Bill','George','Ronald','Jimmy','Gerald','Richard','Lyndon','John','Dwight','Harry','Franklin','Herbert','Calvin','Warren','Woodrow'],
         'StartYear' : [2009,2001,1993,1989,1981,1977,1974,1969,1963,1961,1953,1945,1933,1929,1923,1921,1913],
}

dfPresidents = pd.DataFrame(data=pres)
for president in dfPresidents.iterrows():

    #print(president['Name'])
    print(president[1]['Name'] + ' ' + str(NationalNames[NationalNames['Name'] == president[1]['Name']][NationalNames['Gender'] == 'M'][NationalNames['Year'] == president[1]['StartYear']+1].Count.values[0]/NationalNames[NationalNames['Name'] == president[1]['Name']][NationalNames['Gender'] == 'M'][NationalNames['Year'] == president[1]['StartYear']].Count.values[0]*100-100))


