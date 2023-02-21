#!/usr/bin/env python
# coding: utf-8



import pandas as pd
pd.set_option('display.max_colwidth',None)
import numpy as np
np.random.seed(27)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.color_palette('muted')
sns_colours = sns.color_palette()
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import kstest,boxcox,skew
from collections import defaultdict

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb


from mlxtend.plotting import heatmap, plot_learning_curves
from mlxtend.regressor import StackingRegressor




# Import data. We know there is a date column so we will set it to be converted to datetime format
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', parse_dates = ['last_review'])




data.info()




data.describe().T




# Explicitly show the number of NaNs in each column, if any
def nan_count(df):
    for col in df:
        nans = df[col].isna().sum()
        if nans>0:
            print(f'{col} contains {nans} NaNs')

nan_count(data)




# Inspect missing name and host name entries for concurrence
missing_names = set(data[data['name'].isna()]['id'])
missing_hosts = set(data[data['host_name'].isna()]['id'])
missing_names.intersection(missing_hosts)




# Inspect missing review entries for concurrence
missing_last_rev = set(data[data['last_review'].isna()]['id'])
missing_rpm = set(data[data['reviews_per_month'].isna()]['id'])
len(missing_last_rev.intersection(missing_rpm))




# Fill missing values and drop ID column
data['name'].fillna('Information Missing',inplace=True)
data['host_name'].fillna('None',inplace=True)
data['reviews_per_month'].fillna(0,inplace=True)
cleaned_data = data.drop(['id','host_name'],axis=1)




# Reinspect dataframe information
cleaned_data.info()




X_train, X_test, y_train, y_test = train_test_split(cleaned_data.drop(['price'],axis=1),cleaned_data['price'],                                                    test_size=0.1, random_state = 27)

# Recombine training data
train_df = pd.concat([X_train,y_train],axis=1)
print(f'Training data dimensions: {train_df.shape}')
display(train_df.sample())




_ = train_df['price'].hist(bins=500)
_ = plt.title('New York Airbnb Room Prices')
_ = plt.xlabel('Room Price $')

print(f'The skewness of the room price is {np.round(train_df["price"].skew(),2)}')




print(f'There are {len(train_df[train_df["price"]==0])} listings with a price of $0\n')
display(train_df[train_df["price"]==0])




# Drop listings with a price of 0
train_df.drop(train_df[train_df['price']==0].index,inplace=True)
X_test.drop(y_test[y_test==0].index,inplace=True)
y_test.drop(y_test[y_test==0].index,inplace=True)




# Create rugplot to see spread of listings
fig = plt.figure(figsize=(12,2))
_ = sns.rugplot(train_df['price'])
_ = plt.title('Rugplot of New York Airbnb Room Prices')
_ = plt.xlabel('Room Price $')




high_prices = train_df.nlargest(10,'price')
display(high_prices)

prices = [500,1000,2000,3000,4000,5000]
for price in prices:
    perc = round(sum((train_df['price']>price)/len(train_df)),4)*100
    print(f'{perc:.2f}% of listings cost more than ${price}')




# Split out properties over $1000
high_price_lists = train_df.drop(train_df[train_df['price']<1000].index)

# Remove high price properties from other datasets
train_df.drop(train_df[train_df['price']>=1000].index,inplace=True)
X_test.drop(y_test[y_test>=1000].index,inplace=True)
y_test.drop(y_test[y_test>=1000].index,inplace=True)




# Graphical analysis of listing prices >$1000
pic = plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png',0)
fig = plt.figure(figsize=(20,16))
fig.suptitle('New York Airbnb Listings >$1000',fontsize=16,y=0.9)
ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2,colspan=2)
_ = plt.imshow(pic,extent = [min(data['longitude']),max(data['longitude'])                             ,min(data['latitude']),max(data['latitude'])])

# Add subplot for scatterplot with same axis extents as underlying image
_ = ax1.axis([min(data['longitude']),max(data['longitude']),min(data['latitude']),max(data['latitude'])])
_ = sns.scatterplot(x='longitude',y='latitude',data=high_price_lists,hue='price',palette='coolwarm')
_ = ax1.legend().texts[0].set_text('Price $')

ax2 = plt.subplot2grid((2,3),(0,2))
_ = sns.barplot(y=high_price_lists['neighbourhood_group'].value_counts().index,                x = high_price_lists['neighbourhood_group'].value_counts(),ax=ax2,orient='h')
_ = ax2.set_title('Neighbourhood')
_ = ax2.set_xlabel('Number of Listings')
_ = ax2.tick_params(axis='y',rotation=45)

ax3 = plt.subplot2grid((2,3),(1,2))
_ = sns.barplot(y = high_price_lists['room_type'].value_counts().index,
               x = high_price_lists['room_type'].value_counts(),ax=ax3,orient='h')
_ = ax3.set_title('Room Type')
_ = ax3.set_xlabel('Number of Listings')
_ = ax3.tick_params(axis='y',rotation=45)




fig,ax = plt.subplots(1,3,figsize=(18,8))
_ = train_df.hist(column = 'calculated_host_listings_count',bins=327,color = colours[1],alpha=0.8,ax=ax[0])
_ = ax[0].set_title('Histogram of Listings Per Host')
_ = ax[0].set_xlabel('Listings Per Host')
_ = ax[0].set_ylabel('Count of Listings')

_ = train_df.hist(column='calculated_host_listings_count',bins = 323,color = colours[0],alpha = 0.8,ax=ax[1],cumulative=True)
_ = ax[1].set_title('Cumulative Sum of Listings')
_ = ax[1].set_xlabel('Listings Per Host')
_ = ax[1].set_ylabel('Cumulative Listings')

_ = train_df[train_df['calculated_host_listings_count']>10].hist(
            column='calculated_host_listings_count',bins = 94,color = colours[2],alpha=0.8,ax=ax[2])
_ = ax[2].set_title('Listings > 10')
_ = ax[2].set_xlabel('Listings Per Host')
_ = ax[2].set_ylabel('Count of Listings')




print(f'The mean number of listings per host is {np.floor(train_df["calculated_host_listings_count"].mean())}')
print(f'The median number of listings per host is {np.floor(train_df["calculated_host_listings_count"].median())}')
print(f'The modal number of listings per host is {np.floor(train_df["calculated_host_listings_count"].mode()[0])}')
print(f'''\nThe modal number of listings accounts for {round(len(train_df[train_df['calculated_host_listings_count']==1])/len(train_df),2)*100}% of all listings''')




# Look at the average price by the number of listings per host
price_by_number_of_listings = train_df.groupby('calculated_host_listings_count').agg({'price':['mean','median']})
price_by_number_of_listings.rename_axis('Listings Per Host',inplace=True)
price_by_number_of_listings.rename(columns={'mean':'Mean Price','median':'Median Price'},inplace=True)
price_by_number_of_listings.columns = price_by_number_of_listings.columns.droplevel() # Drop top level column index "price"


fig, ax = plt.subplots(2,1,figsize=(16,8))
_ = plt.suptitle('Mean and Median Room Price by Number of Host Listings',fontsize='14',y=0.93)
_ = sns.barplot(x=price_by_number_of_listings.index,y=price_by_number_of_listings['Mean Price'],ax=ax[0])
_ = ax[0].set_ylabel('Mean Room Price ($)')
_ = sns.barplot(x=price_by_number_of_listings.index, y=price_by_number_of_listings['Median Price'],ax=ax[1])
_ = ax[1].set_ylabel('Median Room Price ($)')




listing_numbers = [11,12,49]
fig, ax = plt.subplots(1,3,figsize=(12,8),sharex=True,sharey=True)
_ = plt.suptitle('Boxplot of Room Prices for Hosts with 11, 12 and 49 Listings Each',y=0.93,fontsize='14')
for ind,listing in enumerate(listing_numbers):
    print(f'''There are {len(train_df[train_df["calculated_host_listings_count"]==listing].groupby("host_id"))} different hosts with {listing} listings each''')
    _ = sns.boxplot(train_df[train_df["calculated_host_listings_count"]==listing]['price'],labels=[''],ax=ax[ind],orient='v',                   color=sns_colours[ind])
    _ = ax[ind].set_xlabel(f'{listing} listings per host')
    _ = ax[ind].set_ylabel("")
_ = ax[0].set_ylabel('Room Price ($)')




print(f'There are {len(set(train_df["neighbourhood"]))} unique neighbourhoods in the data set')
print(f'There are {len(set(train_df["neighbourhood_group"]))} unique neighbourhood groups in the data set')




# Create dataframe of number of listings per neighbourhood

neighbourhoods = train_df.groupby('neighbourhood').agg({'neighbourhood':'count'})
neighbourhoods.rename(columns={'neighbourhood':'Count'},inplace=True)
neighbourhoods = neighbourhoods.sort_values(by='Count',ascending=False)




# Plot cumulative distribution of listings across neighbourhoods and number of listings in top 5 neighbourhoods
fig,ax = plt.subplots(1,2,figsize=(16,6))

y = np.arange(len(neighbourhoods['Count']))/len(neighbourhoods['Count'])
x = np.sort(neighbourhoods['Count'])
_ = ax[0].plot(x,y,marker='.',color = sns_colours[1],linestyle='none',alpha = 0.8)
_ = ax[0].set_title("Empirical CDF of Listings per Neighbourhood")
_ = ax[0].set_xlabel("Number of Listings per Neighbourhood")
_ = ax[0].set_ylabel("ECDF")

_ = sns.barplot(x = neighbourhoods.index[:5],y = neighbourhoods['Count'][:5],ax=ax[1]).set_title('Number of Listings in Top 5 Neighbourhoods')
_ = ax[1].set_ylabel('Number of Listings')
_ = ax[1].set_xlabel('')
_ = ax[1].tick_params(axis='x',labelrotation=-15)

print(f'The mean number of listings per neighbourhood is {np.floor(neighbourhoods["Count"].mean())}')
print(f'The median number of listings per neighbourhood is {neighbourhoods["Count"].median()}\n')




# Create underlying image using map of new york
pic = plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png',0)
fig,ax = plt.subplots(1,figsize=(14,14))

# Set extends to correspond to coordinate system
_ = plt.imshow(pic,extent = [min(data['longitude']),max(data['longitude']),min(data['latitude']),max(data['latitude'])])

# Add subplot for scatterplot with same axis extents as underlying image
_ = ax.axis([min(data['longitude']),max(data['longitude']),min(data['latitude']),max(data['latitude'])])

# Create colourmap that doesn't have grey in it
colrs = ['lightcoral','goldenrod','yellowgreen','steelblue','mediumpurple']

_ = sns.scatterplot(x = 'longitude',y = 'latitude',hue = 'neighbourhood_group',data = train_df,                   alpha = 0.7, palette = colrs)
_ = ax.legend().texts[0].set_text('Neighbourhood Area')




# Create dataframe of rooms in each neighbourhood group
rooms_in_ngroup = train_df.groupby('neighbourhood_group').agg({'name':'count'})

# Create data frame without highest price rooms for plotting
price_less_500 = train_df[train_df['price']<500].sort_values(by='neighbourhood_group')

fig, ax = plt.subplots(1,2,figsize=(12,6))
_ = sns.barplot(x=rooms_in_ngroup.index,y=rooms_in_ngroup['name'],ax=ax[0],orient='v')
_ = plt.suptitle('Analysis of Room Information in Different Neighbourhood Groups',y=0.94)
_ = ax[0].set_xlabel("")
_ = ax[0].set_ylabel('Number of Rooms')

_ = sns.violinplot(x = 'neighbourhood_group',y='price',data = price_less_500,ax=ax[1])
_ = ax[1].set_xlabel("")
_ = ax[1].set_ylabel('Room Price (\$) for rooms <$500')




log_data = train_df[['host_id','longitude','latitude']]
log_data['Log Price'] = np.log(train_df['price'])#+abs(min(np.log(train_df['price'])))
log_data['Log Reviews per Month'] = np.sqrt(train_df['reviews_per_month'])




# Plot listing locations with room price as a colourmap
fig, ax = plt.subplots(1,1,figsize=(9,9))
# Set extends to correspond to coordinate system
_ = plt.imshow(pic,extent = [min(data['longitude']),max(data['longitude']),                             min(data['latitude']),max(data['latitude'])])

# Add subplot for polygons with same axis extents as underlying image
_ = ax.axis([min(data['longitude']),max(data['longitude']),min(data['latitude']),             max(data['latitude'])])

_ = sns.scatterplot(x='longitude',y='latitude',data=log_data,hue='Log Price',                    palette = 'coolwarm',alpha=0.5).set_title('Location of New York listings under $500')




# Plot breakdown of room types and how they vary per neighbourhood
room_types_by_neighbourhood = pd.crosstab(train_df['neighbourhood_group'],train_df['room_type'],                                          normalize='index')

fig, ax = plt.subplots(1,2,figsize=(14,4))

_ = room_types_by_neighbourhood.plot.bar(stacked=True,ax=ax[1])
_ = ax[1].legend(title='Room Type').texts[0]
_ = ax[1].set_title('Room Type Proportions by Neighbourhood')
_ = ax[1].set_xlabel("")
_ = ax[1].tick_params(axis='x',rotation=360)

_ = sns.barplot(x=train_df['room_type'].value_counts().index,y = train_df['room_type'].value_counts(),ax=ax[0])
_ = ax[0].set_title('Breakdown of Room Types')
_ = ax[0].set_ylabel("Number of Rooms")




# Check median price of each room type
room_types = train_df['room_type'].unique()
for room in room_types:
    med_price = round(np.median(train_df['price'][train_df['room_type']==room]),0)
    print(f'The median price of a {room} is ${med_price}')




# Look at spread of minimum nights
fig = plt.figure(figsize=(12,2))
_ = sns.rugplot(train_df['minimum_nights']).set_title('Minimum Nights for Booking')




train_df[train_df['minimum_nights']>365]




# Look at scatter of prices against minimum nights
fig, ax = plt.subplots(1,2,figsize=(14,6),sharey=True)
plt.suptitle('Room Price vs Minimum Nights')

# Set up dataframe excluding bookings longer than one year
train_df2 = train_df[train_df['minimum_nights']<367]

# Add plot axes
_ = sns.scatterplot(x='minimum_nights',y='price',data=train_df,ax=ax[0]).set_title('All Listings')
_ = ax[0].set_ylabel('Room Price $')
_ = ax[0].set_xlabel('Minimum Nights')

_ = _ = sns.scatterplot(x='minimum_nights',y='price',data=train_df2,ax=ax[1],                       color = sns_colours[1]).set_title('Listings of One Year or Less')
_ = ax[1].set_xlabel('Minimum Nights')




# Look at last review date by year
train_df['review_year'] = train_df['last_review'].dt.year

fig = plt.figure(figsize=(10,5))
_ = sns.distplot(train_df['review_year'],kde=False,bins=[2011.5,2012.5,2013.5,2014.5                                                         ,2015.5,2016.5,2017.5,2018.5,2019.5]).set_title('Last Review Date by Year')
_ = plt.xlabel('Year')
_ = plt.ylabel('Number of Reviews')




# Look at number of reviews in each neighbourhood group
reviews_in_neighbourhood = train_df.groupby('neighbourhood_group').agg({'name':'count','number_of_reviews':'sum'})
reviews_in_neighbourhood['ratio'] = reviews_in_neighbourhood['number_of_reviews']/reviews_in_neighbourhood['name']
plt.figure(figsize=(8,6))
_ = sns.barplot(x = reviews_in_neighbourhood.index,y = reviews_in_neighbourhood['ratio'],orient='v')
_ = plt.title('Review Ratio by Neighbourhood')
_ = plt.xlabel("")
_ = plt.ylabel('Mean Reviews per Listing')
_ = plt.tick_params(axis='x',rotation = 360)




# Look at reviews per room type
room_type_reviews = train_df.groupby('room_type').agg({'name':'count','number_of_reviews':'sum'})
room_type_reviews['ratio'] = room_type_reviews['number_of_reviews']/ room_type_reviews['name']
fig = plt.figure(figsize=(6,6))
_ = sns.barplot(x = room_type_reviews.index,y=room_type_reviews['ratio'],orient='v')
_ = plt.title('Review Ratio by Room Type')
_ = plt.xlabel("")
_ = plt.ylabel("Mean Reviews per Listing")




train_df['has_review'] = train_df['number_of_reviews']>0
print('Mean listing price split by whether the listing has been reviewed \n')
has_review = train_df.groupby('has_review').agg({'price':'mean'})
display(has_review)
print('\nProportion of room types split by whether a listing has been reviewed \n')
display(pd.crosstab(train_df['has_review'],train_df['room_type'],normalize='index'))




# Set up bootstrap test for mean price difference
difference = has_review.loc[False] - has_review.loc[True]

# Find size of samples to draw
no_reviews_size = sum(train_df['has_review']==False)
reviews_size = sum(train_df['has_review']==True)

# Initialize bootstrap variables
reps = 1000
differences = []

for i in range(reps):
    sample1 = train_df.sample(no_reviews_size,replace=True)
    sample2 = train_df.drop(sample1.index)
    
    sample1_price = np.mean(sample1['price'])
    sample2_price = np.mean(sample2['price'])
    differences.append(sample1_price-sample2_price)




# Review results of bootstrapping
fig = plt.figure(figsize=(8,6))
_ = plt.hist(differences, bins=50,color=colours[2])
_ = plt.title('Difference in Mean Price Between Bootstrap Samples')
_ = plt.xlabel('Price Difference $')
_ = plt.ylabel('Count')

print(f'''The price difference between listings with and without reviews is ${round(difference[0],1)}.
The maximum price difference in {reps} boostrap samples was ${round(max(map(abs,differences)),1)}''')




# Extract time on site from total reviews and reviews per month - this will not work for listings with no reviews
train_df['months_on_site'] = train_df['number_of_reviews']/train_df['reviews_per_month']
train_df['months_on_site'].fillna(0,inplace=True)
fig = plt.subplots(figsize=(12,8))
x = train_df[train_df['months_on_site']>0]['months_on_site']
scatter = sns.scatterplot(x=x,y='number_of_reviews',data=train_df,hue='room_type',)

x2 = np.arange(0,max(train_df['months_on_site']))
y = x2

line, = plt.plot(x2,y,color=colours[5],linewidth=3,label='One Review per Month')

_ = plt.title('Listings With at Least One Review')
_ = plt.xlabel('Months Listing on Airbnb')
_ = plt.ylabel('Total Number of Reviews')
_ = plt.legend().texts[1].set_text('Room Type')

perc = sum(train_df['reviews_per_month']>1)
at_least_1 = sum(train_df['number_of_reviews']>0)
print(f'{perc/at_least_1*100:.1f}% of listings with at least one review average more than one review per month\n')
print(f'{perc/train_df.shape[0]*100:.1f}% of all listings average more than one review per month\n')




top_availabilities = train_df['availability_365'].value_counts().to_frame().head(10)
top_availabilities.rename({'availability_365':'Count'},axis='columns',inplace=True)
top_availabilities.rename_axis('Availability (days per year)',inplace=True)

display(top_availabilities)




# Look at how mean availability varies with room type
room_avail = train_df.groupby('room_type').agg({'availability_365':'mean'})
display(room_avail)




# Reimport Data
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', parse_dates = ['last_review'])
data.columns




# Clean and split data using methods identified during EDA
def clean_and_split(df):
    '''Returns cleaned and split data: training, testing'''
    temp = df.copy()
    temp = temp.drop(['id','host_name'],axis=1)
    temp['name'] = temp['name'].fillna('Information Missing')
    temp['reviews_per_month'] = temp['reviews_per_month'].fillna(0)
    temp = temp.drop(temp[temp['price']==0].index)
    temp = temp.drop(temp[temp['price']>1000].index)
    
    X_tr,X_te,y_tr,y_te = train_test_split(temp.drop(['price'],axis=1),temp['price'],                                           test_size=0.1,random_state=27)
    
    training = pd.concat([X_tr,y_tr],axis=1)
    testing = pd.concat([X_te,y_te],axis=1)
    
    return training, testing
    




# Apply cleaning function to get training and test data set
train_df,test_df = clean_and_split(data)




print(f'The skewness of the listing price is {train_df["price"].skew():.2f}')
log_price = np.log(train_df['price'])
print(f'\nThe skewness of the logarithm of the price is {log_price.skew():.2f}')




# Select only numeric features
numeric_df = train_df.select_dtypes(include=['int64','float64'])
numeric_df.drop(['host_id','longitude','latitude'],axis=1,inplace=True)

# Create correlation matrix
correlation = numeric_df.corr()

hm = heatmap(correlation.values,column_names=correlation.columns,row_names=correlation.columns,            figsize=(6,6),column_name_rotation=90)




print(f'''The skewness of the host listings count is {train_df["calculated_host_listings_count"].skew():.2f}''')

log_listings = np.log(train_df['calculated_host_listings_count'])
box_listings = boxcox(train_df['calculated_host_listings_count'])

print(f'''The skewness of the log of the host listings count is {skew(log_listings):.2f}''')

print(f'''The skewness of the boxcox transform of the host listings count is {skew(box_listings[0]):.2f} using a lambda of {box_listings[1]:.3f}''')

print(f'\nThe skewness of the availability is {train_df["availability_365"].skew():.2f}')




def create_model_df(df):
    temp = df.copy()
    temp['log_price'] = np.log(temp['price'])
    temp = pd.get_dummies(temp,columns=['room_type','neighbourhood_group'])
    temp['has_review'] = temp['number_of_reviews']>0
    temp['calculated_host_listings_count'] = boxcox(temp['calculated_host_listings_count'],lmbda=-1.298)
    temp = temp.drop(['name','host_id','neighbourhood','latitude','longitude','number_of_reviews',                     'last_review','reviews_per_month','price'],axis=1)
    
    return temp




visible_df = create_model_df(train_df)
testing_df = create_model_df(test_df)

visible_df.head()




# Split the visible data into training and validation sets
X_train,X_valid,y_train,y_valid = train_test_split(visible_df.drop('log_price',axis=1),                                                  visible_df['log_price'],test_size=0.2,random_state=27)

training_df = pd.concat([X_train,y_train],axis=1)
validation_df = pd.concat([X_valid,y_valid],axis=1)




# Create correlation matrix
correlation = training_df.corr()

hm = heatmap(correlation.values,column_names=correlation.columns,row_names=correlation.columns,            figsize=(12,12),column_name_rotation=90)




# Select model features to use as estimators
estimators = training_df.columns.to_list()
estimators.remove('log_price')




# Construct and evaluate naive model
mean_price = training_df['log_price'].mean()
naive_preds = training_df['log_price']-mean_price

# Find rmse and convert from log price to price in $
naive_rmse = (mean_squared_error(np.exp(training_df['log_price']),np.exp(naive_preds))**0.5)
print(f'The RMSE of the naive model is ${naive_rmse:.2f}')




# Instantiate Base Models
lin_reg = LinearRegression()
svr = SVR()
lasso = Lasso()
xgbr = xgb.XGBRegressor()
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()

models = [('Linear Regression',lin_reg),('Support Vector',svr),('Lasso',lasso),         ('XGBoost',xgbr),('Decision Tree',dt),('Random Forest',rf)]




# Use CV socring with neg mean squared error to assess base models. For speed use only 10000 training examples
scores = defaultdict(list)
for name, model in models:
    cv = cross_val_score(model,X=training_df.loc[:10000,estimators],y = training_df.loc[:10000,'log_price'],                        scoring='neg_mean_squared_error',verbose=1,cv=5)
    scores[name] = cv.mean()

display(pd.DataFrame(pd.Series(scores),columns=['MSE']))




# Obtain RMSE score for model
lin_reg.fit(training_df[estimators],training_df['log_price'])
lin_reg_preds = lin_reg.predict(testing_df[estimators])
lin_reg_rmse = (mean_squared_error(np.exp(testing_df['log_price']),np.exp(lin_reg_preds))**0.5)

print(f'The RMSE of the Linear Regression model is ${lin_reg_rmse:.3f}')




# Plot learning curves
fig = plt.figure(figsize=(12,12))
_ = plot_learning_curves(training_df[estimators], training_df['log_price'], testing_df[estimators],                     testing_df['log_price'], lin_reg,scoring='mean_squared_error')
_ = plt.ylim(0.21,0.23)




# XGBoost
xgbr_params = {'objective':['reg:squarederror'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [3,4,5],
              'subsample': [0.5,0.7,0.8],
              'colsample_bytree': [0.6,0.7,0.8],
              'n_estimators': [100,200]}

xgbr_gs = RandomizedSearchCV(xgbr,param_distributions= xgbr_params,n_iter = 100,scoring='neg_mean_squared_error',                             cv=3,verbose=0)
xgbr_gs.fit(validation_df[estimators],validation_df['log_price'])
print(f'The lowest MSE achieved is: {xgbr_gs.best_score_:.3f} with parameters:{xgbr_gs.best_params_}')




xgbr_final = xgb.XGBRegressor(**xgbr_gs.best_params_)
xgbr_final.fit(training_df[estimators],training_df['log_price'])
xgbr_preds = xgbr_final.predict(testing_df[estimators])
xgbr_rmse = (mean_squared_error(np.exp(testing_df['log_price']),np.exp(xgbr_preds))**0.5)

print(f'The RMSE of the XGBoost model is ${xgbr_rmse:.3f}')




# Plot learning curves
fig = plt.figure(figsize=(12,12))
_ = plot_learning_curves(training_df[estimators], training_df['log_price'], testing_df[estimators],                     testing_df['log_price'], xgbr_final,scoring='mean_squared_error')




# Add columns for actual price, predicted price and absolute error to dataframe.
testing_df['price'] = np.exp(testing_df['log_price'])
testing_df['predicted_price'] = np.exp(xgbr_preds)
testing_df['error'] = abs(testing_df['price']-testing_df['predicted_price'])




# Look at scatterplot and histogram of errors.
fig,ax = plt.subplots(1,figsize=(12,6))
_ = testing_df.hist('error',bins=100,ax=ax)
_ = ax.set_title('Absolute Prediction Errors of XGB Regressor')
_ = ax.set_xlabel('Abs Error ($)')
_ = ax.set_ylabel('Count')

fig,ax1 = plt.subplots(1,figsize=(12,6))
_ = sns.scatterplot(x='price',y='error',data=testing_df,ax=ax1,color=sns_colours[1])
_ = ax1.set_title('Absolute Errors of XGB Regressor')
_ = ax1.set_xlabel('Room Price ($)')
_ = ax1.set_ylabel('Abs Error ($)')

