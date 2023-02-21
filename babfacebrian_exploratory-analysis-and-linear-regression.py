#!/usr/bin/env python
# coding: utf-8



import pandas as pd 
import numpy as np
from __future__ import unicode_literals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns




# Reading in the data
kick_data = pd.read_csv("../input/most_backed.csv")
del kick_data['Unnamed: 0']




#rename some columns (not needed, I just used this data with other data with similar column names)
kick_data.rename(columns = {'amt.pledged' : 'pledged_amount'}, inplace=True)
kick_data.rename(columns = {'pledge.tier' : 'pledge_tier'}, inplace=True)
kick_data.rename(columns = {'num.backers' : 'backers'}, inplace=True)
kick_data.rename(columns = {'num.backers.tier' : 'backers_by_tier'}, inplace=True)
kick_data.rename(columns = {'blurb' : 'pitch'}, inplace=True)




#4000 of the most funded campaigns that were on Kickstarter.
kick_data.head()




kick_data.describe()




kick_data.corr()




kick_data.category.value_counts(ascending = False)[0:10]




#Total $ amount pledged by category. 

amt_pledged = kick_data['pledged_amount'].groupby(kick_data['category'])
pledged = amt_pledged.sum().sort_values(ascending=0)[0:10]


ax = pledged.plot(kind="bar")
ax.set_title("Amount Pledged by Category")
ax.set_ylabel("Amount Pledged")
ax.set_xlabel("Category")
vals = ax.get_yticks()




# Amount pledged by location.

amt_locations = kick_data['pledged_amount'].groupby(kick_data['location'])
top_10_locations = amt_locations.sum().sort_values(ascending=0)[0:10]

ax = top_10_locations.plot(kind="bar")
ax.set_title("Pledged Amount by Market \n (top 10 markets)")
ax.set_ylabel("Amount Pledged")
ax.set_xlabel("Market")
vals = ax.get_yticks()




#Amount of backers by category 

backers = kick_data['backers'].groupby(kick_data['category'])
back_cat = backers.sum().sort_values(ascending=0)[0:10]


ax = back_cat.plot(kind="bar")
ax.set_title("Backers by Categroy \n (top 10)")
ax.set_ylabel("# of backers")
ax.set_xlabel("Category")
vals = ax.get_yticks()




# amount of backers by location.

backed_locations = kick_data['backers'].groupby(kick_data['location'])
top_10_locations = backed_locations.sum().sort_values(ascending=0)[0:10]

ax = top_10_locations.plot(kind="bar")
ax.set_title("Backers by market \n (top 10 markets)")
ax.set_ylabel("# of Backers")
ax.set_xlabel("Market")
vals = ax.get_yticks()




#keep only the columns we need for linear regression.

kick_train = kick_data.drop(['pitch','by','currency','location','backers_by_tier','pledge_tier', 'url'], axis=1)




train_cols = ['title', 'category', 'backers', 'goal', 'pledged_amount']
kick_train = kick_train.reindex(columns= train_cols)




kick_train['percent_funded'] =(kick_train.pledged_amount / kick_train.goal)

for value in kick_train.percent_funded:
    np.around(value, decimals=1)
    
kick_train.head()




kick_train.corr()




sns.lmplot('backers', 'pledged_amount', kick_train)




log_columns = ['backers','pledged_amount','percent_funded']
log_kick = kick_train.copy()
log_kick[log_columns] = log_kick[log_columns].apply(np.log10)




sns.lmplot('backers', 'pledged_amount', log_kick)




sns.lmplot('percent_funded', 'pledged_amount', log_kick)




sns.lmplot('percent_funded', 'backers', log_kick)




log_kick.plot(kind='scatter', x='backers', y='pledged_amount')




log_kick.plot(kind='scatter', x='percent_funded', y='pledged_amount')




log_kick.plot(kind='scatter', x='percent_funded', y='backers')




from sklearn import feature_selection, linear_model
import statsmodels.api as sm
from sklearn.model_selection import train_test_split




def plot_category(category):
    kick_test = kick_train.loc[kick_data['category'] == category]
    kick_test.plot(kind='scatter', x='backers', y='pledged_amount')
    X = kick_test[["backers"]]
    Y = kick_test["pledged_amount"]




# a function to fit a regression on each category

def ols_reg(category):
    kick_test = kick_train.loc[kick_data['category'] == category]
    X = kick_test[["backers"]]
    Y = kick_test["pledged_amount"]
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.summary())
kick_train.category.unique()




c = 'Video Games'
ols_reg(c)
plot_category(c)




X = log_kick[["backers"]]
y = log_kick["pledged_amount"]

#Spliting testing and train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


# Fiting the linear model
model = linear_model.LinearRegression()
results = model.fit(X_train, y_train)

# Print the coefficients
print (results.intercept_, results.coef_)




model = sm.OLS(y_train, X_train)
results = model.fit()

print(results.summary())




test_model = sm.OLS(y_test, X_test)
test_results = model.fit()

print (test_results.summary())




def get_linear_model_metrics(X, Y, algo):
    pvals = feature_selection.f_regression(X, Y)[1]
   
    algo.fit(X,Y)
    
    predictions = algo.predict(X)
    plt.scatter(predictions, Y)
    plt.show()
    print 'Predictions: ' + str(predictions)
    residuals = (Y-algo.predict(X)).values
    print 'Residual: ' + str(residuals)

    print 'P Values:', pvals
    print 'Coefficients:', algo.coef_
    print 'y-intercept:', algo.intercept_
    print 'R-Squared:', algo.score(X,Y)
    plt.figure()
    plt.hist(residuals, bins=np.ceil(np.sqrt(len(Y))))
    
    
    return algo

backers_X = X_test
amt_Y = y_test
lm = linear_model.LinearRegression(fit_intercept=False)
lm = get_linear_model_metrics(backers_X, amt_Y, lm)

# I do not know why this is giving me a syntax error, this works fine in my python notebook, if you have an idea let me know! 




kick_live = pd.read_csv("../input/live.csv")
del kick_live['Unnamed: 0']




live_data = kick_live.drop(['blurb','by','currency','location','state','type','url'], axis=1)

train_cols = ['title', 'amt.pledged','percentage.funded']
live_data = kick_live.reindex(columns= train_cols)




live_data.head(10)




live_data.sort_values(by='amt.pledged', ascending=0)




funded_pro = live_data.loc[live_data['percentage.funded'] >= 100]
funded_pro.sort_values(by='percentage.funded', ascending=0)




log_columns_live = ['amt.pledged','percentage.funded']
live_log = live_data.copy()
live_log[log_columns_live] = live_log[log_columns_live].apply(np.log10)




live_log.plot(kind='scatter', x='percentage.funded', y='amt.pledged')




log_kick.plot(kind='scatter', x='percent_funded', y='pledged_amount')




P = log_kick[["percent_funded"]]
b = log_kick["pledged_amount"]

#Spliting testing and train.
P_train, P_test, b_train, b_test = train_test_split(P, b, test_size=0.5, random_state=0)


# Fiting the linear model
model_2 = linear_model.LinearRegression()
results = model_2.fit(P_train, b_train)

# Print the coefficients
print (results.intercept_, results.coef_)




lm = linear_model.LinearRegression(fit_intercept=False)
lm = get_linear_model_metrics(P_test, b_test, lm)




Lx = kick_live[['percentage.funded']]
Ly = kick_live['amt.pledged']

#Spliting testing and train.
Lx_train, Lx_test, Ly_train, Ly_test = train_test_split(P, b, test_size=0.5, random_state=0)


# Fiting the linear model
model_3 = linear_model.LinearRegression()
results = model_3.fit(P_train, b_train)

# Print the coefficients
print (results.intercept_, results.coef_)




livemodel = sm.OLS(Ly_train, Lx_train)
results = livemodel.fit()

print(results.summary())






