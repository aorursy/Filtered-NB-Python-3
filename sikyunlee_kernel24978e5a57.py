#!/usr/bin/env python
# coding: utf-8



# Data manipulation
import pandas as pd
import numpy as np

# More Data Preprocessing & Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




la_df = pd.read_csv('../input/detail-listingscsv/Detail_listings.csv') #data that will be used for analysis and modeling




## Tip: transposing the data when using head() function helps to see all of the columns vertically instead of horizontally
pd.options.display.max_rows = la_df.shape[1]
la_df.head(2).T




## To look at all of the column names
la_df.columns




## List of the variables that will be removed 
## Leftover variables that will be used in the analysis

drop = ['listing_url', 'scrape_id', 'last_scraped', 'summary', 'space', 'description', 'neighborhood_overview',
        'notes', 'transit', 'access', 'interaction', 'house_rules', 'thumbnail_url', 'medium_url', 'picture_url',
        'xl_picture_url', 'host_id', 'host_url', 'host_about', 'host_thumbnail_url', 'host_picture_url',
        'calendar_updated', 'calendar_last_scraped', 'license', 'name', 'host_name', 'zipcode', 'id','city', 'state',
        'market','jurisdiction_names', 'host_location', 'street', 'experiences_offered','country_code','country',
        'has_availability','host_neighbourhood','neighbourhood_cleansed','neighbourhood_group_cleansed','smart_location',
        'neighbourhood','host_acceptance_rate','square_feet']




## drop the variables that needs to be removed
df = la_df.drop(drop, axis=1)




## df will be used for analysis from now on.
df.info()




# Before dropping duplicates
df.shape




## Dropping any duplicate rows
df = df.drop_duplicates()




# After dropping duplicates 
df.shape




## Create a dataframe of only categorical variables 
cat_df = df.select_dtypes(['object', 'bool'])


## Create a dataframe of only numerical variables 
num_df = df.select_dtypes(['int', 'float', 'int64', 'float64'])




## Check if the above computation has worked correctly and if num_df only has numerical data type columns
num_df.head().T




# Check the cat_df
cat_df.head().T





# Getting all the features that should be numerical, but are typed as objects (strings)
cat_to_num = ['host_response_rate', 'price', 'weekly_price',
              'monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people']

# Keeping changes in a temporary copied DataFrame
cat_to_num_df = cat_df[cat_to_num].copy(deep=True)




## Check the new deep copied data
cat_to_num_df.head(2)




## Remove the percent sign, then convert to a number 
cat_to_num_df['host_response_rate'] = cat_to_num_df['host_response_rate'].str.replace('%','').astype(float)/100




## Columns that needs the '$' removed and converted to float type
price_cols = ['price', 'weekly_price','monthly_price', 'security_deposit', 'cleaning_fee', 'extra_people']




## For each of the price columns, remove commas and dollar signs, then convert it to float using a for loop
for col in price_cols:
    cat_to_num_df[col] = cat_to_num_df[col].str.replace('$', '').str.replace(',', '')
    cat_to_num_df[col] = cat_to_num_df[col].astype(float)
    




## Check the converted column data
cat_to_num_df.head(2)




## Append the converted column data into the num_df
num_df = pd.concat([num_df, cat_to_num_df], axis=1)




## Drop the old column data from the cat_df since these columns were converted and appended to num_df
cat_df = cat_df.drop(cat_to_num, axis=1)




bi_cols = []


## Create a for loop to check if each column stored only 2 unique values in its columns. The columns that have only 2 unique values are considered as Binary (or Boolean) 
## The Binary (Boolean) columns will be stored in the bi_cols list

for col in cat_df.columns:
    if cat_df[col].nunique() == 2:
        bi_cols.append(col)

## Check if computation was done correctly
cat_df[bi_cols].head()




## Convert all binary columns to 1's and 0's. 
for col in bi_cols:
    cat_df[col] = cat_df[col].map({'f': 0, 't': 1})
    




## Check the bi_cols in cat_df using head to see if computation was done correctly
cat_df[bi_cols].head(2)




## Unpack these values in order for them to be meaningful
cat_df[['host_verifications', 'amenities']].head(2)




## Define a method to use .strip() method in a loop for all x values in l variable
def striplist(l):
    '''
    To be used with the apply method on a packed feature
    '''
    return([x.strip() for x in l])




## Replace unnecessary characters with blanks and split strings with ',' 
# Note: Lines can be broken into next lines using "\" to improve readability
cat_df['host_verifications'] = cat_df['host_verifications'].str.replace('[', '')                                                            .str.replace(']', '')                                                            .str.replace("'",'')                                                            .str.lower()                                                            .str.split(',')                                                            .apply(striplist)




## Perform same method for amenities column
cat_df['amenities'] = cat_df['amenities'].str.replace('{', '')                                          .str.replace('}', '')                                          .str.replace('"','')                                          .str.lower()                                          .str.split(',')                                          .apply(striplist)




## Define MLB
mlb = MultiLabelBinarizer()




## Use the MultiLabelBinarizer to fit and transform host_verifications
## Store this result in an object called host_verif_matrix
host_verif_matrix = mlb.fit_transform(cat_df['host_verifications'])




## Check the output after using the transformer
host_verif_df = pd.DataFrame(host_verif_matrix, columns = mlb.classes_)
host_verif_df.head(2)




## Use the MultiLabelBinarizer to fit and transform amenities
amenities_matrix = mlb.fit_transform(cat_df['amenities'])

## Store this result in a DataFrame called amenities_df (similar to what we did above with host_verif_df)
amenities_df = pd.DataFrame(amenities_matrix, columns = mlb.classes_)

## heck the output after using the transformer
amenities_df.head(2)




## Drop the blank named column
amenities_df = amenities_df.drop([''], axis=1)




## Drop the old host_verifications and amenities features from cat_df
cat_df = cat_df.drop(['host_verifications', 'amenities'], axis=1)

## Concatenate amenities_df and host_verif_df to the original cat_df DataFrame
cat_df = pd.concat([cat_df, amenities_df, host_verif_df], axis=1)




## Put the date columns into dt_cols
dt_cols = ['host_since', 'first_review', 'last_review']

## Check for computation
cat_df[dt_cols].head(1)




## Convert these columns to the datetime format 
for col in dt_cols:
    cat_df[col] = pd.to_datetime(cat_df[col], infer_datetime_format=True)




##  Capture today's date using to_datetime (this will be the standard to be subtracted from)
today = pd.to_datetime('today')




## Create one new date feature that counts number of days since today's date for each of the three date features
for col in dt_cols:
    num_df[col+'_days'] = (today - cat_df[col]).apply(lambda x: x.days)




## Check if computation was done correctly (ignore the NaN for now)
num_df[[dt_col+'_days' for dt_col in dt_cols]].head(10)




## Drop the original date columns from cat_df
cat_df = cat_df.drop(dt_cols, axis=1)




## Concatenate the num_df and cat_df into one new DataFrame named cleaned_df
cleaned_df = pd.concat([num_df, cat_df], axis=1)




# Creating a discrete feature based on how recent the last review was
bins = [0, 365, 730, 1095, 1470, np.inf]
labels = ['last year', 'last 2 years','last 3 year', 'last 4 years', 'more than 4 years']
cleaned_df['last_review_discrete'] = pd.cut(num_df['last_review_days'], bins=bins, labels=labels)

# Filling the Null values in this new column with "no reviews", assuming Null means there are no reviews
cleaned_df['last_review_discrete'] = np.where(cleaned_df['last_review_discrete'].isnull(),
                                              'no reviews', 
                                              cleaned_df['last_review_discrete'])




## Copy the data to eda
eda = cleaned_df[:]




## Check what values are in room_type
eda['room_type'].value_counts()




## Save the room type's proportion covered by Entire home/apt and Private room into eda_viz
eda_viz = eda[eda['room_type'].isin(['Entire home/apt', 'Private room'])]




## Visualize the price distribution 
plt.figure(figsize=(10,4))
sns.distplot(eda['price'])
plt.show()




## To get rid of outliers, filtering out prices that are greater than 3 sample standard deviations from the mean
## Normally 3 standard deviations means that any data points outside of this range is outside of the 95% Confidence Interval Level
price_mean = np.mean(eda['price'])
price_std = np.std(eda['price'])
price_cutoff = price_mean + price_std*3




## Get rid of the prices exceeding average + 3*std_dev
eda_viz = eda_viz[eda_viz['price'] < price_cutoff]




## Visualize the prices in respect to room type and superhost
fgrid = sns.FacetGrid(eda_viz, col='room_type', height=6,)
fgrid.map(sns.boxplot, 'last_review_discrete', 'price', 'host_is_superhost', 
          order=labels, hue_order = [0,1])

for ax in fgrid.axes.flat:
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set(xlabel=None, ylabel=None)

l = plt.legend(loc='upper right')
l.get_texts()[0].set_text('Is not Superhost')
l.get_texts()[1].set_text('Is Superhost')

fgrid.fig.tight_layout(w_pad=1)




## Assume we want to predict the price feature. 
## If price is the variable we want to predict, then we have to disregard rows that don't have it
## Drop all rows with missing values in the 'price' column
cleaned_df = cleaned_df.dropna(subset=['price'])




## Calculating proportion by summing NA values and dividing by length of DF
prop_na = cleaned_df.isna().sum()/len(cleaned_df)
## Filtering out columns with less than 5% NA values to clean up the visualization below
prop_na_05 = prop_na[prop_na > 0.05]
prop_na_05 = prop_na_05.sort_values(0, ascending=True).reset_index()




## Plotting proportion of NA values for all columns where more than 5% is missing. 
plt.figure(figsize=(6, 5))

barh = plt.barh(prop_na_05['index'], prop_na_05[0], alpha=0.85, color='skyblue')
for i in range(3):
    i += 1
    barh[-i].set_color('darkblue')

plt.title('Proportion of NA Values')
plt.vlines(x=.3, ymin=0, ymax=20, color='darkred', linestyles='dashed')
plt.xticks(np.arange(.1, 1.01, .1))

plt.tight_layout()




## Drop the 3 missing columns identified above from cleaned_df
drop_na_cols = ['monthly_price', 'weekly_price', 'security_deposit']
drop.extend(drop_na_cols)
cleaned_df = cleaned_df.drop(drop_na_cols, axis=1)




## Create a temporary column called "sum_na_row" in cleaned_df that contains the number of NA values per row
cleaned_df['sum_na_row'] = cleaned_df.isna().sum(axis=1)




## Use matplotlib or seaborn to plot the distribution of this new column. 

plt.figure(figsize=(8,5))
sns.distplot(cleaned_df['sum_na_row'], bins=15, kde=False)
plt.xticks(np.arange(0, 21, 5))
plt.title('Distribution of Amount of Missing Values per Row')
plt.annotate('This is strange.\n Why are there so many missing values here?',
             xy=(8.0,100),
             xytext=(1,1100),
             arrowprops={'arrowstyle':'->'})
plt.show()




## Filter cleaned_df for only rows with 10 or more missing values. Store this in a temporary DataFrame
temp = cleaned_df[cleaned_df['sum_na_row'] >= 10]

## Get the names of the columns that contain missing values from this temporary DF
na_cols = temp.columns[temp.isna().any()]

# Check if computation was done correctly and transpose for column readability
temp[na_cols].transpose()




# Collecting the numerical review-related columns
zero_fill_cols = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
               'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
               'review_scores_value', 'reviews_per_month', 'first_review_days', 'last_review_days']




## Isolate all categorical columns (i.e. columns of dtype 'object'), 
## Visualize count or barplots for each column to see how these features are distributed to check for frequent mode features
cat_cols = cleaned_df.select_dtypes(['object']).columns.values




plt.figure(figsize=(13, 10))
i = 1
for col in cat_cols:
    plt.subplot(3, 3, i)
    
    sns.countplot(cleaned_df[col])
    plt.xticks(rotation=90)
    plt.tick_params(labelbottom=True)
    
    i += 1

plt.tight_layout()




# Getting indices of columns that still contain missing values
columns_idxs_missing = np.where(cleaned_df.isna().any())[0]

# Getting the names of these columns
cols_missing = cleaned_df.columns[columns_idxs_missing]

# Taking a peek at what's left
cleaned_df[cols_missing].head()




## Drop the sum_na_row feature we made from cleaned_df since this is not needed any longer
cleaned_df = cleaned_df.drop(['sum_na_row'], axis=1)




## Check for other columns that have not been transformed or examined and fill them with the median values
    
features_accounted_for = np.concatenate([cat_cols,np.array(zero_fill_cols+['price'])])
all_cols = cleaned_df.columns.values

median_fill_cols = np.setdiff1d(all_cols, features_accounted_for)




## Store cleaned_df without the price column in a variable called X. 
X = cleaned_df.drop('price', axis=1)

## Store cleaned_df['price'] in a variable called y
y = cleaned_df['price']


## Split the data using train_test_split using a train_size of 80% and test size of 20%
## Store all these in the variables below
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)




# Setting this option to None to suppress a warnings
pd.options.mode.chained_assignment = None




# impute using a for loop
for col in cat_cols:
    imputer = SimpleImputer(strategy='most_frequent')
    # fit the impute to X_training data
    imputer.fit(X_train[[col]])

    
    # using the transform method to fill NA values with the most frequent value, then updating our DFs
    X_train[col] = imputer.transform(X_train[[col]])
    X_test[col] = imputer.transform(X_test[[col]])




## Impute the zero_fil_cols features using an imputer with strategy = "constant" and fill_value = 0
for col in zero_fill_cols:
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    imputer.fit(X_train[[col]])
    
    X_train[col] = imputer.transform(X_train[[col]])
    X_test[col] = imputer.transform(X_test[[col]])




## Impute the median_fill_cols using an imputer with strategy = "median"
for col in median_fill_cols:
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train[[col]])
    
    X_train[col] = imputer.transform(X_train[[col]])
    X_test[col] = imputer.transform(X_test[[col]])




## In X_train, create a new feature called "capacity_to_beds" by dividing the "accomodates" feature by "beds"
X_train['capacity_to_beds'] = X_train['accommodates']/X_train['beds']

## Do the same thing for X_test. 
X_test['capacity_to_beds'] = X_test['accommodates']/X_test['beds']

## Check for correct computation
X_train[['accommodates', 'beds','capacity_to_beds']].head()




## TO DO: Fill infinite values with zero in X_train and X_test. (Hint: use np.where and np.isinf can be helpful)
X_train['capacity_to_beds'] = np.where(np.isinf(X_train['capacity_to_beds']),
                                          0,
                                          X_train['capacity_to_beds'])

X_test['capacity_to_beds'] = np.where(np.isinf(X_test['capacity_to_beds']),
                                          0,
                                          X_test['capacity_to_beds'])




plt.figure(figsize=(12,4))

# Creating plot on the left
plt.subplot(121)
sns.distplot(X_train['cleaning_fee'])
plt.title('Before Log-Transform')

# Creating plot on the right
plt.subplot(122)
log_transform_train = np.where(np.isinf(np.log(X_train['cleaning_fee'])), 0, np.log(X_train['cleaning_fee']))
log_transform_test = np.where(np.isinf(np.log(X_test['cleaning_fee'])), 0, np.log(X_test['cleaning_fee']))
sns.distplot(log_transform_train)
plt.title('After Log-Transform')

plt.show()




## Update the "cleaning_fee" features in X_train and X_test with their log-transformed values
X_train['cleaning_fee'] = log_transform_train
X_test['cleaning_fee'] = log_transform_test




temp_df = X_train.select_dtypes(['float', 'int'])

# Gathering binary features
bi_cols = []
for col in temp_df.columns:
    if temp_df[col].nunique() == 2:
        bi_cols.append(col)




## Store all the columns we need to standardize in cols_to_standardize.
cols_to_standardize = np.setdiff1d(temp_df.columns, bi_cols)




## Instantiate the StandardScaler() 
scaler = StandardScaler()
## Fit the scaler to X_train's cols_to_standardize only since binary (0,1) columns do not need to be standardized
scaler.fit(X_train[cols_to_standardize])

## Transform X_train and X_test's cols_to_standardize and update the DataFrames
X_train[cols_to_standardize] = scaler.transform(X_train[cols_to_standardize])
X_test[cols_to_standardize] = scaler.transform(X_test[cols_to_standardize])




## Check for number of unique values in each categorical columns
print('Unique Values per categorical column : ')
for col in cat_cols:
    print(f'{col}: {X_train[col].nunique()}')




## OHE the categorical column data
for col in cat_cols:
    ## Instantiate the OneHotEncoder 
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    ## Fit ohe to the current column "col" in X_train
    ohe.fit(X_train[[col]])
    
    # This extracts the names of the dummy columns from OHE
    dummy_cols = list(ohe.categories_[0])
    
    # This creates new dummy columns in X_train and X_test that will be filled
    for dummy in dummy_cols:
        X_train[dummy] = 0
        X_test[dummy] = 0
    
    ## Transform the X_train and X_test column "col" and update the dummy_cols created above
    X_train[dummy_cols] = ohe.transform(X_train[[col]])
    X_test[dummy_cols] = ohe.transform(X_test[[col]])




## Drop the original cat_cols from X_train and X_test
X_train = X_train.drop(cat_cols, axis=1)
X_test = X_test.drop(cat_cols, axis=1)




threshold = 0.8

# Calculating an absolute value correlation matrix
corr_mat = X_train.corr().abs()

# Getting upper triangle of this matrix only
upper = pd.DataFrame(np.triu(corr_mat, k=1), columns=X_train.columns)

# Select columns with correlations above threshold
corr_col_drop = [col for col in upper.columns if any(upper[col] > threshold)]

print(f'There are {len(corr_col_drop)} columns to remove out of {len(X_train.columns)}.')




X_train = X_train.drop(corr_col_drop, axis=1)
X_test = X_test.drop(corr_col_drop, axis=1)




from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error




rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
svr = SVR()

models = [rf, gbr, svr]




results = []

for model in models:
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_preds)
    mae = mean_absolute_error(y_test, y_preds)

    
    metrics = {}
    metrics['model'] = model.__class__.__name__
    metrics['mse'] = mse
    metrics['mae'] = mae
    results.append(metrics)




pd.set_option('display.float_format', lambda x: '%7.2f' % x)

pd.DataFrame(results, index=np.arange(len(results))).round(50)






