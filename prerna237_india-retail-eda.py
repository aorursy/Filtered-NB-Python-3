#!/usr/bin/env python
# coding: utf-8



'''
Tasks to finish
1. Plot the series (Done)
1a. Plot trendlines (Done)
2. extract seasonality (Done)
3. Read articles about regression techniques 
4. Finalize and prep data for the new model,
5. Model
6. Results finetuning methods
7. Plot results
'''




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

# suppress all warnings
import warnings
warnings.filterwarnings('ignore')




#Read the dataset and printing some vital stats
my_raw_data=pd.read_csv("../input/India_Key_Commodities_Retail_Prices_1997_2015.csv")
print(my_raw_data.info())




my_raw_data["Date"]=my_raw_data["Date"].apply(lambda x: str(pd.Period(x,freq='Y')))




print(my_raw_data.describe())
print(my_raw_data["Country"].unique())




# To analyse the growth of prices of a commodity over years
commodity_list=my_raw_data["Commodity"].unique()
for eachCommodity in commodity_list:
    plt.figure(figsize=(10,10))
    sorted_df=my_raw_data[my_raw_data["Commodity"]==eachCommodity]
    sorted_df=pd.DataFrame(sorted_df.groupby("Date").mean()).reset_index()
    plt.plot(sorted_df["Date"],sorted_df["Price per Kg"],marker='o',label=eachCommodity)
    plt.ylabel('Price per Kg')
    plt.xlabel('Year')
    plt.legend(loc='top')
    plt.margins(0.1)
    plt.show()

#Also plotting all in one plot to contrast them 
# To analyse the growth of prices of a commodity over years
commodity_list=my_raw_data["Commodity"].unique()
for eachCommodity in commodity_list:
    plt.figure(figsize=(10,10))
    sorted_df=my_raw_data[my_raw_data["Commodity"]==eachCommodity]
    sorted_df=pd.DataFrame(sorted_df.groupby("Date").mean()).reset_index()
    plt.plot(sorted_df["Date"],sorted_df["Price per Kg"],marker='o',label=eachCommodity)
    plt.ylabel('Price per Kg')
    plt.xlabel('Year')
    plt.legend(loc='top')
    plt.margins(0.1)
plt.show()




# Plotting the trendlines to explore a bit more
for eachCommodity in commodity_list:
    sorted_df=my_raw_data[my_raw_data["Commodity"]==eachCommodity]
    sorted_df=pd.DataFrame(sorted_df.groupby("Date").mean()).reset_index()
    vals=np.polyfit(pd.to_numeric(sorted_df["Date"]),pd.to_numeric(sorted_df["Price per Kg"]),1)
    func=np.poly1d(vals)
    plt.plot(func(pd.to_numeric(sorted_df["Date"])),marker='o',label=eachCommodity)
    plt.ylabel('Price per Kg')
    plt.xlabel('Year')
    plt.legend(loc='top')
    plt.margins(0.1)
plt.show()




#Another method of visualization would be a bar graph of %change over the years
xvals=list()
yvals=list()
for eachCommodity in commodity_list:
    sorted_df=my_raw_data[my_raw_data["Commodity"]==eachCommodity]
    sorted_df=sorted_df.groupby("Date").mean()
    perc=((sorted_df["Price per Kg"].max())-(sorted_df["Price per Kg"].min())/sorted_df["Price per Kg"].min())
    xvals.append(eachCommodity)
    yvals.append(perc)
bar_width = 0.35
opacity = 0.4
plt.bar(xvals,yvals, bar_width,
                alpha=opacity,
                label='Percentage change')
plt.legend(loc='top')
plt.xticks(xvals, rotation='vertical')
plt.margins(0.1)
plt.show()




#To analyze which ciy has the highest cost of living
agg_df=pd.DataFrame(my_raw_data.groupby(['Date','Centre','Commodity']).mean()).reset_index()
commodity_list=my_raw_data["Commodity"].unique()
date_list=my_raw_data["Date"].unique()
centre_list=my_raw_data["Centre"].unique()
for eachYear in date_list:
    for eachCentre in centre_list:
        temp=agg_df.loc[(agg_df["Date"]==eachYear) & (agg_df["Centre"]=="AIZWAL")]
        if(temp["Commodity"].unique()!=commodity_list):
            print(eachCentre+' has ',temp["Commodity"].unique(),' in ',eachYear)




#To analyze which commodities cost higher during certain months of the year
month_df=pd.read_csv("../input/India_Key_Commodities_Retail_Prices_1997_2015.csv")
month_df["Date"]=month_df["Date"].apply(lambda x: str(pd.Period(x,freq='M').month))




commodity_list=month_df["Commodity"].unique()
aggregate_df=pd.DataFrame(month_df.groupby(['Date','Commodity']).mean()).reset_index()
aggregate_df




a=aggregate_df[(aggregate_df["Date"]=='9') & (aggregate_df["Commodity"]=='Tomato')]["Price per Kg"]
print(a.values[0])




xvals=list()
yvals=list()
bar_width = 0.35
opacity = 0.4
for eachCommmodity in commodity_list:
    for eachNum in range(1,12):
        xvals.append(eachNum)
        yvals.append((aggregate_df[(aggregate_df["Date"]==str(eachNum)) & (aggregate_df["Commodity"]==eachCommmodity)]["Price per Kg"]).values[0])
    labelStr='Prices of '+eachCommmodity
    plt.bar(xvals,yvals, bar_width,
                label=labelStr)
    plt.legend(loc='top')
    plt.margins(0.1)
    plt.show()




#To figure out the K best
input_data=pd.read_csv("../input/India_Key_Commodities_Retail_Prices_1997_2015.csv",parse_dates=[0])
print("Shape before:",input_data.shape)
#filtering out outliers
input_data=input_data[np.abs(input_data["Price per Kg"]-input_data["Price per Kg"].mean()) <= (3*input_data["Price per Kg"].std())]
print("Shape after:",input_data.shape)




print(type(input_data["Date"]))




input_data.loc[0]["Date"].month




input_data["Year"]=input_data["Date"].apply(lambda x: str(x.year))
input_data["Month"]=input_data["Date"].apply(lambda x: str(x.month))
input_data["Day"]=input_data["Date"].apply(lambda x: str(x.day))




input_data
del input_data["Date"]




#removing null values if any
input_data = input_data.dropna(subset=['Price per Kg'])
#list of categorical
categorical_value_cols=[]
for eachCol in input_data.columns:
    if(input_data[eachCol].dtype=='object'):
        categorical_value_cols.append(eachCol)
        
#removing country attribute as country is same across all data points
categorical_value_cols.remove('Country')
print("Categorical value cols=",categorical_value_cols)
input_data_X = input_data[categorical_value_cols].copy(deep=True)
input_data_X




from sklearn.preprocessing import LabelEncoder
centre_enc = LabelEncoder()
comm_enc = LabelEncoder()
reg_enc=LabelEncoder()
day_enc=LabelEncoder()
month_enc=LabelEncoder()
year_enc=LabelEncoder()
input_data_X["Centre"] = centre_enc.fit_transform(input_data_X["Centre"])
input_data_X["Commodity"] = comm_enc.fit_transform(input_data_X["Commodity"])
input_data_X["Region"]=reg_enc.fit_transform(input_data_X["Region"])
input_data_X["Day"]=day_enc.fit_transform(input_data_X["Day"])
input_data_X["Month"]=month_enc.fit_transform(input_data_X["Month"])
input_data_X["Year"]=year_enc.fit_transform(input_data_X["Year"])




# print (input_data_X.info())
input_data_X




input_data_Y=input_data["Price per Kg"]
print(input_data_X.shape)
print(input_data_Y.shape)




clubbed_df= pd.concat([input_data_X,input_data_Y], axis=1, sort=False)
print(clubbed_df.info())




from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
selector=SelectKBest(score_func=f_regression,k='all')
selector.fit(input_data_X,input_data_Y)
print("scores_:",selector.scores_)
scores = selector.scores_
# Plotting the best features
fig=plt.figure(figsize=(12,5))
plt.bar(np.arange(len(scores)),scores,label = input_data_X.columns)
plt.axhline(y=max(scores)*0.6,color='b',linewidth=.5)
plt.xticks(np.arange(len(scores)),input_data_X.columns,rotation="vertical")
plt.ylabel("K-Scores")
plt.show()
    




list_of_features=list(input_data_X.columns)
# list_of_features.remove('Price per Kg')
list_of_features.remove('Centre')
list_of_features.remove('Month')
list_of_features.remove('Day')
print((list_of_features))
input_data_X=input_data_X[list_of_features]




from sklearn.linear_model import SGDRegressor 
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(input_data_X, input_data_Y)
print((test_y.values))




classif=SGDRegressor(early_stopping=True,)
classif.fit(train_X,train_y.values,)




y_pred=classif.predict(test_X)




def mean_absolute_percentage_error(y_true, y_pred): 
    y_trueVals=y_true.values
    print(y_true,y_pred)
#     print(np.mean([1,2,3]))
#     print("Error=",np.mean(np.abs((y_trueVals - y_pred)/y_trueVals)))
    return np.mean(np.abs(((y_trueVals - y_pred) / y_trueVals))) * 100




print(mean_absolute_percentage_error(test_y.values,y_pred))









df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
            'two'],
                   'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'baz': [1, 2, 3, 4, 5, 6],
                   'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
print(df)




df.pivot(index='foo', columns='bar', values='zoo')




test_csv=pd.read_csv("../input/India_Key_Commodities_Retail_Prices_1997_2015.csv")
sample=test_csv.iloc[150000:170000,:]
print(sample.describe())
print(sample['Commodity'].unique())
(sample[sample['Centre']=='GUWAHATI'])




# sample.reset_index()
sample["Date"]=sample["Date"].apply(lambda x: str(pd.Period(x,freq='Y')))
# sample[sample['Centre']=='PATNA']
sample




agg=(sample.groupby(['Date','Centre','Commodity']).mean())
# agg.to_csv('aggregate_values.csv', sep='\t')
agg




new_agg=pd.DataFrame(agg).reset_index()
new_agg
# new_agg[new_agg['Centre']=='AGARTALA']
# print(new_agg['Date'].values)




print(sample["Price per Kg"].max())
print(sample["Price per Kg"].min())




print(sample.groupby(["Date","Centre"]).sum())




sample.set_index("Date")
print(sample.info())
print(sample)




dummy_df=sample.groupby("Date").mean()




dummy_df.reset_index()
print(dummy_df)




plt.plot(dummy_df["Price per Kg"],dummy_df.index.values,linestyle='--', marker='o', color='b')
plt.show()









plt.plot([1,2,3,4], [1,4,9,16],[1,2,3,4], [1,14,19,14])
plt.show()









commodity_list=my_raw_data["Commodity"].unique()
print(commodity_list)
for eachCommodity in commodity_list:
    sorted_df=my_raw_data[my_raw_data["Commodity"]==eachCommodity]
    sorted_df=sorted_df.groupby("Date").mean()
    plt.plot(sorted_df["Price per Kg"],sorted_df.index.values,marker='o')




plt.show()




columns=list(my_raw_data.columns)
columns.remove('Price per Kg')
print(columns)
print(my_raw_data.describe(include=['object']))





# print(tomato_prices)
# dates=my_raw_data[my_raw_data["Commodity"]=="Tomato"]["Date"]
# # extract avg tomato prives per year
# years=dates.apply(lambda x: pd.Period(x,freq='Y'))
# print(years.size)
# print(tomato_prices.size)
# new_df=pd.concat([years, tomato_prices], axis=1)
# mean_df=new_df.groupby("Date").mean()
# print(mean_df.head())
# plt.plot(tomato_prices,dates)
# plt.show()




#Separating out data for each commodity 
# factors_list=list(columns)
# factors_list.remove('Commodity')
for eachFactor in factors_list:
    for eachCommodity in commodity_list:
        plt.scatter(x=my_raw_data[my_raw_data["Commodity"]==eachCommodity][eachFactor],y=my_raw_data[my_raw_data["Commodity"]==eachCommodity]["Price per Kg"])
        plt.xlabel=eachFactor
        plt.ylabel="Price per Kg of "+eachCommodity
        plt.title=""+eachFactor+" vs Price per kg of "+eachCommodity
        plt.show()
    
# plt.scatter(x=my_raw_data[my_raw_data["Commodity"]=="Sugar"]["Region"],y=my_raw_data[my_raw_data["Commodity"]=="Sugar"]["Price per Kg"])
# plt.show()
# plt.scatter(x=my_raw_data[my_raw_data["Commodity"]=="Tomato"]["Region"],y=my_raw_data[my_raw_data["Commodity"]=="Tomato"]["Price per Kg"])
# plt.show()









#Making scatter plot to visualize any correlation between various factors and Price









plt.scatter(my_raw_data["Commodity"],my_raw_data["Price per Kg"])
plt.show()









my_raw_data.boxplot(column="Price per Kg", by="Region",figsize=(8, 8))

plt.suptitle("")
    plt.show()




plt.matshow(my_raw_data.corr())
plt.xticks(range(len(my_raw_data.columns)), my_raw_data.columns)
plt.yticks(range(len(my_raw_data.columns)), my_raw_data.columns)
plt.colorbar()
plt.show()
my_raw_data

