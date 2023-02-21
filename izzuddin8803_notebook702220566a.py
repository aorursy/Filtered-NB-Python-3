#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns #seaborn is already installed
import matplotlib.pyplot as plt









#loading the dataset
df = pd.read_csv('../input/data.csv',encoding="ISO-8859-1",dtype={'CustomerID': str,})
print(df.shape)
df.head(3)




# Check any duplicated data set and column formats
print(sum(df.duplicated(keep="first")),"transaction rows affected")
df.dtypes




#remove duplicate rows from dataset, reformatting columns and defining new fields for data exploration
df=df.drop_duplicates()
df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format="%m/%d/%Y %H:%M")
df.StockCode=df.StockCode.str.upper() #to see if can separate some more
df.rename(columns={"InvoiceDate":'InvoiceDateTime'}, inplace=True)
df['InvoiceDate'] = pd.to_datetime([dt.datetime.date(d) for d in df['InvoiceDateTime']]) #to extract date only from datetime info
#df['InvoiceTime'] = df['InvoiceDateTime'].dt.time #to extract time only from datetime info
df['mth_end_dt'] = df['InvoiceDate']+pd.offsets.MonthEnd(0) #to get month end date position
df.shape




#summary of the numeric and object columns
print(df.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))
df.describe(include=[np.object])




#We count the negative value of of quantity and Unit Price
print("The number of rows with negative Quantity:",sum(n < 0 for n in df.Quantity))
print("The number of rows with negative UnitPrice:",sum(n < 0 for n in df.UnitPrice))




#Count Unique value in all dataset columns
df.nunique()




#visualising Quantity
#will someone consider modifying this into boxplot?

plt.figure(figsize=(15,15))

x=df.Quantity.value_counts().reset_index().as_matrix().transpose()
plt.subplot(411) #1st digit #rows, 2nd digit #columns, 3rd digit plot number
plt.scatter(x[0], x[1], marker='o')
plt.title('Quantity plots',fontsize=15)
plt.ylabel('Occurrence',fontsize=12)

x=df[df['Quantity'].abs()<20000].Quantity.value_counts().reset_index().as_matrix().transpose()
plt.subplot(412)
plt.scatter(x[0], x[1], marker='o')
plt.ylabel('Occurrence')

#Based on 99th percentile
x=df[df['Quantity'].abs()<100].Quantity.value_counts().reset_index().as_matrix().transpose()
plt.subplot(413)
plt.scatter(x[0], x[1], marker='o')
plt.ylabel('Occurrence',fontsize=12)

#Based on 3rd quartile
x=df[df['Quantity'].abs()<10].Quantity.value_counts().reset_index().as_matrix().transpose()
plt.subplot(414)
plt.scatter(x[0], x[1], marker='o')
plt.xlabel('Quantity',fontsize=12)
plt.ylabel('Occurrence',fontsize=12)

plt.show()




#Identifying what is the equivalent counterpart while taking a look at the outliers.
df[df['Quantity'].abs()>60000]





#1.Quantity outliers due to customer making mistake in their order, and it has been cancelled. Can be removed from the #dataset. 
#2.The mistake order is offset by another transaction, given a different invoicedate and invoiceno. The only common field #shared is the stockcode, customerID and quantity ordered. 
#3. Invoices cancelled will have the letter C in front of the 6 digit invoiceno. 




#Identifying what is the equivalent counterpart while taking a look at the outliers.
df[(df['Quantity'].abs()>5000) & (df['Quantity'].abs()<20000)]




#Check how NaN values affect the dataset
print('Number of rows in each column affected by existence of non-existing values:')
df.isnull().sum()




#Now lets check what is in our negative quantity
df[df["Quantity"]<=0].head(10)




#Access all the NaN element in the Description discovered earlier when checking number of rows affected with missing values
#from IPython.display import display, HTML
#dfNADescription=df[df.Description.isnull()]
print('Descriptive statistics of numeric columns:\n',df[df.Description.isnull()].describe())
print('\nDescriptive statistics of CustomerID columns:\n',df[df.Description.isnull()].CustomerID.describe())




#plot price to see outliers
#since describe reveal that min and max quantities are in the range >10000 as compared to most sections of the dataset

plt.figure(figsize=(15,10))

x=df.UnitPrice.value_counts().reset_index().as_matrix().transpose()
plt.subplot(411)
plt.scatter(x[0], x[1], marker='o')
plt.title('UnitPrice plots',fontsize=15)
plt.ylabel('Occurrence',fontsize=12)

x=df[df['UnitPrice'].abs()<20000].UnitPrice.value_counts().reset_index().as_matrix().transpose()
plt.subplot(412)
plt.scatter(x[0], x[1], marker='o')
plt.ylabel('Occurrence',fontsize=12)

#99th-percentile
x=df[df['UnitPrice'].abs()<18].UnitPrice.value_counts().reset_index().as_matrix().transpose()
plt.subplot(413)
plt.scatter(x[0], x[1], marker='o')
plt.ylabel('Occurrence',fontsize=12)

#3rd quartile
x=df[df['UnitPrice'].abs()<4.13].UnitPrice.value_counts().reset_index().as_matrix().transpose()
plt.subplot(414)
plt.scatter(x[0], x[1], marker='o')
plt.ylabel('Occurrence',fontsize=12)
plt.xlabel('UnitPrice',fontsize=12)

plt.show()




#Identifying what is the equivalent counterpart while taking a look at the outliers.
df[df['UnitPrice'].abs()>10000]




#The stockcodes aren't linked to any other item purchases - single item per invoice.
df[df['InvoiceNo']=='537632']




#Investigating stockcode and invoiceno fields based on discoveries made

#defining the variables
df['length_stockcode']=df.StockCode.str.len()
df['length_invoiceno']=df.InvoiceNo.str.len()
df['invoiceno_letter1']=df['InvoiceNo'].str[0]

print("length of InvoiceNo:\n",df.length_invoiceno.value_counts(sort=True)      .reset_index(name='no_rows').rename(columns={'index':'length of InvoiceNo'}))
print("\nFirst letter for invoice:\n",df.invoiceno_letter1.value_counts(sort=True)      .reset_index(name='no_rows').rename(columns={'index':'invoice first letter'}))
print("\nCross table first letter for invoice against invoiceno length:\n"      ,pd.crosstab(df['invoiceno_letter1'],df['length_invoiceno'],margins=True))
print("\nlength of StockCode:\n",df.length_stockcode.value_counts(sort=True)      .reset_index(name='no_rows').rename(columns={'index':'length of StockCode'})) 





#Discoveries: 
#1.Invoice length is either 6 or 7; those with length 7 will start with letter A or C 
#2.Since most rows have StockCode of length 5 or 6, this is considered the legitimate StockCode referring to item description. 
#4.StockCode length is between 1 to 12 (excl. 10 and 11)




#display what are the unique stockcodes for invoiceno with length below 5 or above 8
print("\nDescription for StockCode with length below 5 or above 8 and number of lines affected:")
      #\n"\
#      ,df[(df['length_stockcode']<5) | (df['length_stockcode']>8)]\
#      [['length_stockcode','invoiceno_letter1','StockCode','Description']]\
#      .groupby(by=['invoiceno_letter1','length_stockcode','StockCode']).Description.value_counts().reset_index(name='Freq'))

df[(df['length_stockcode']<5) | (df['length_stockcode']>8)][['length_stockcode','invoiceno_letter1','StockCode','Description']].groupby(by=['length_stockcode','StockCode','invoiceno_letter1']).Description.value_counts().reset_index(name='Freq')




#analysing stockcode and invoiceno field

#display what are the unique stockcodes for invoiceno of length 7 or 8
print("\nDescription for StockCode with length 7 or 8 and number of lines affected:\n")
df[(df['length_stockcode']==7) | (df['length_stockcode']==8)][['length_stockcode','invoiceno_letter1','StockCode','Description']].groupby(by=['length_stockcode','StockCode']).Description.value_counts().reset_index(name='freq')




#From initial assessment, we found out most problematic description contain lower case letter and '?' symbol
df[df['Description'].str.contains("^[a-z]|\\?",case=True, na=False)].drop(['mth_end_dt','length_stockcode','length_invoiceno','invoiceno_letter1'],axis=1).head(20)




print('no of rows affected:',df[(df.CustomerID.isnull()) & (df['UnitPrice']==0)].shape[0])
df[(df.CustomerID.isnull()) & (df['UnitPrice']==0)].Description.value_counts().reset_index(name='freq').rename(columns={'index':'Description'}).head(20)




#perform modification on df1
df1=df.copy()
df1.shape




#1. Separate missing description rows
df.NAdesc=df1[(df1.Description.isnull())]
print(df.NAdesc.shape)
df.NAdesc.to_csv('No descriptions.csv',index=False)
df1=df1[~(df1.Description.isnull())]
print(df1.shape)




#2. Separate length-stockcode<5, StockCode=['AMAZONFEE','BANK CHARGES']
df.otherdesc=df1[(df1['length_stockcode']<5) | (df1['StockCode']=='AMAZONFEE')                        | (df1['StockCode']=='BANK CHARGES')]
print(df.otherdesc.shape)
df.otherdesc.to_csv('Other descriptions.csv',index=False)
df1=df1[~((df1['length_stockcode']<5) | (df1['StockCode']=='AMAZONFEE')                        | (df1['StockCode']=='BANK CHARGES'))]
print(df1.shape)




#3. Remove rows that contain problematic description and UnitPrice=0 into a new file.
df.weird=df1[(df1.CustomerID.isnull()) & (df1['UnitPrice']==0)]
print(df.weird.shape)
df.weird.to_csv('weird description and or unitprice.csv',index=False)
df1=df1[~((df1.CustomerID.isnull()) & (df1['UnitPrice']==0))]
print(df1.shape)




#4. to remove Cancelled transactions into a separate dataset
df.Cancel=df1[df1['invoiceno_letter1']=='C']
print(df.Cancel.shape)
df.Cancel.to_csv('Cancelled Transactions v2.csv',index=False)
df1=df1[df1['invoiceno_letter1']!='C']
print(df1.shape)
#we have 9251 row of cancelled transaction




#1. Assign value to our missing customerID based on the Invoice Number
df1["CustomerID"].fillna("R"+df1["InvoiceNo"], inplace=True)
df.CustomerID.value_counts(sort=True)
NewID = df1.groupby(['CustomerID','Description']).sum()
NewID.head()




#Check if there is any lowercase letter in stock code
df1[df1['StockCode'].str.contains("[a-z]",case=True, na=False)]
#No lowercase code




#Now let check of negative Quantity
df1[df1["Quantity"]<=0]
#Our negative quantity is zero as most of our negative quantity are inside Nan Description that we removed earlier




#we check if our problematic Description got removed by the drop negative and zero unit price
from IPython.display import display, HTML
df1.prob=df1[df1['Description'].str.contains("^[a-z]|\\?",case=True, na=False)]
df1.prob1=df1.prob.Description.str.split(expand=True).stack().value_counts().to_frame().reset_index()
HTML(df1.prob1.to_html())
#Our hypothesis is true as all the problematic description also got removed by our previous droping actvities




#Now let check if the result of our cleaning process
df1.describe()
# We no longer have negative value for our numeric column




df1.isnull().sum()
#All our Nan have been removed from the description




#Check if our new customer ID that been assigned with "R" + InvoiceNo
df1[df1['CustomerID'].str.contains("R",case=False, na=False)]




# Check again for any Nan Value
df1.isnull().sum()




df1.nunique()




df1.StkDsc=df1.groupby("StockCode")['Description'].nunique().reset_index()
df1stk=df1.StkDsc[df1.StkDsc["Description"]>1]
df1stk.head()
#We have 215 row of stock code that have more than 1 description




import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import seaborn as sns
plotly.tools.set_credentials_file(username='JLLam', api_key='rHULPcQZPrG6MrCNj0VC')
plotly.tools.set_config_file(world_readable=True,sharing='public')




df2 = df1.copy()
df2["Revenue"]=df2["Quantity"]*df2["UnitPrice"]




#Calculte the total Revenue thorught out the year
df2.Revenue.sum()




#Produce total revenue by months by descending value
df2.groupby(df2['InvoiceDate'].dt.strftime('%B %Y'))['Revenue'].sum().sort_values()




#Group the data set by days of month
df2.groupby(pd.Grouper(key='InvoiceDate', freq='D'))['Revenue'].sum().reset_index().sort_values('InvoiceDate')




#Visualize visualize Revenue by Month
temp = df2.loc[:,('InvoiceDate','Revenue')]
temp.InvoiceDate = df2.InvoiceDate.dt.to_period('M')
temp = temp.groupby(['InvoiceDate'])['Revenue'].sum()
temp = temp.reset_index(drop = False)


plt.figure(figsize=(15,5))
plt.bar(np.arange(len(temp['InvoiceDate'])), temp['Revenue'], align='center', alpha=0.5)
plt.xticks(np.arange(len(temp['InvoiceDate'])), temp['InvoiceDate'])
plt.ylabel('Total Revenue',fontsize=14)
plt.xlabel('Year-Month',fontsize=14)
plt.title('Total Revenue by Month',fontsize=15)
 
plt.show()




#Now group the revenue by week
df2.RevWeek=df2.groupby(pd.Grouper(key='InvoiceDate', freq='W-MON'))['Revenue'].sum().reset_index().sort_values('InvoiceDate')
df2.RevWeek.head()




# Now we combine our weekly revenue line chart with weekly revenue bar chart for better visualization
#On the 4th week, there was no transaction
Revenue1=df2.RevWeek.Revenue
Time1 =df2.RevWeek.InvoiceDate
df2.RevWeek.plot(kind='bar', title ="Total Revenue By Week", figsize=(30, 20), legend=True, fontsize=20)
plt.title('Total Revenue by Weeks',fontsize=30)
plt.xlabel("Week",fontsize=25)
plt.ylabel("Revenue",fontsize=25)
df2.RevWeek['Revenue'].plot(secondary_y=True,color="#8b0000")
plt.show()
#we found out that our sales start to surge on week 41 onwards




#We are going to investigate sales by days of the month
df_Dmonth=df2.copy()
df_Dmonth["day"]=df_Dmonth["InvoiceDate"].dt.day

df_Dmontha=df_Dmonth.groupby("day")["Revenue"].sum()
df_Dmontha.plot(kind='bar', title ="V comp", figsize=(30, 20), legend=True, fontsize=20)
plt.title('Total Revenue by Days of Month',fontsize=30)
plt.ylabel('Total Revenue', fontsize=30)
plt.xlabel('Day of the Month', fontsize=30)
plt.show()
# we concluded that the sale surge at the earlu of the monthand  dwindle as it reach the end of month




#To visualize Revenue by Day of the week
temp = df2.loc[:,('InvoiceDate','Revenue','CustomerID','InvoiceNo')]
temp2 = temp.groupby(temp['InvoiceDate'].dt.weekday_name)['Revenue'].sum().sort_values()
temp2 = temp2.reset_index(drop = False).rename(columns={'InvoiceDate':'Day'})


plt.figure(figsize=(20,10))
plt.bar(np.arange(len(temp2['Day'])), temp2['Revenue'], align='center', alpha=0.5)
plt.xticks(np.arange(len(temp2['Day'])), temp2['Day'])
plt.ylabel('Total Revenue')
plt.xlabel('Day')
plt.title('Total Revenue by Day')
 
plt.show()
#Tuesday and thursday show the highest revenue, No revenue on Saturday

