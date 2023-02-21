#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
pd.options.display.max_columns = 999
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('../input/movie_IMDB.csv')




data.head()




data.shape




data.columns




data.info()




data.isnull().sum().sort_values(ascending = False)[:5]




data.dropna(how = 'any',axis = 0,inplace = True)




data.shape




numerical_features = data.select_dtypes(exclude=['object']).columns
categorical_features = data.select_dtypes(include=['object']).columns




num_data = data[numerical_features]
cat_data = data[categorical_features]




num_data.shape,cat_data.shape




num_data.isnull().sum().sort_values(ascending = False)




num_data.head()




num_data.drop(['title_year'],axis = 1, inplace = True)




numerical_features = numerical_features.drop('title_year')




numerical_features.size




plt.figure(figsize=(8,6))
plt.scatter(range(num_data.shape[0]), np.sort(num_data.imdb_score.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('IMDB_SCORE', fontsize=12)
plt.show()




plt.figure(figsize=(12,8))
sns.distplot(num_data.imdb_score.values, bins=50, kde=False)
plt.xlabel('imdb_score', fontsize=12)
plt.show()




num_data.columns.tolist()




fig, ax = plt.subplots(figsize=(20,10), ncols=2, nrows=3)
sns.set_style("whitegrid")
#sns.boxplot(x="imdb_score", data=num_data,orient = 'v',ax = ax[0][0])
sns.boxplot(x="num_critic_for_reviews", data=num_data,orient = 'v',ax = ax[0][0])
sns.boxplot(x="duration", data=num_data,orient = 'v',ax = ax[0][1])
#sns.boxplot(x="director_facebook_likes", data=num_data,orient = 'v',ax = ax[1][0])
sns.boxplot(x="gross", data=num_data,orient = 'v',ax = ax[1][0])
sns.boxplot(x="num_voted_users", data=num_data,orient = 'v',ax = ax[1][1])
#sns.boxplot(x="num_user_for_reviews", data=num_data,orient = 'v',ax = ax[2][0])
#sns.boxplot(x="title_year", data=num_data,orient = 'v',ax = ax[2][1])
sns.boxplot(x="movie_facebook_likes", data=num_data,orient = 'v',ax = ax[2][1])




num_data.isnull().sum().sort_values(ascending = False)




# from pandas.plotting import scatter_matrix
# scatter_matrix(num_data, alpha=0.2, figsize=(10, 10), diagonal='kde')




num_data.median()




num_data.fillna(num_data.median(),inplace = True)




# most correlated features
import seaborn as sns
corrmat = num_data.corr()
plt.figure(figsize = (10,7))
# or fig, ax = plt.subplots(figsize=(20, 10))
top_corr_features = corrmat.index[abs(corrmat["imdb_score"])>0.1]
g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#g = sns.heatmap(corrmat,annot=True,cmap="RdYlGn")




corrmat.sort_values(["imdb_score"], ascending = False, inplace = True)
print(corrmat.imdb_score)




corrmat.index[abs(corrmat['imdb_score']) > 0.3].tolist()




import numpy as np

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))




# outliers = num_data[(num_data['num_voted_users'] >200000) | (num_data['num_voted_users'] < 1000)].index
# outliers.size

# num_data.drop(outliers,inplace = True)
# cat_data.drop(outliers,inplace = True)

# sns.distplot(num_data['num_voted_users'])




test = outliers_iqr(num_data['imdb_score'])




test = list(test)




num_data.drop(num_data.index[test],inplace = True)
cat_data.drop(cat_data.index[test],inplace = True)




a = num_data[(num_data.num_voted_users < 10000)].index




num_data.drop(a,inplace = True)
cat_data.drop(a,inplace = True)




num_data.shape




# c=np.setxor1d(num_data.index.values,test)

# # c=np.intersect1d(num_data.index.values,test)
# num_data = num_data.loc[c,:]




df_genres = pd.DataFrame(cat_data['genres'])
df_genres = pd.DataFrame(df_genres.genres.str.split('|').tolist(),columns = ["Genre_"+str(i) for i in  range(0,8)] )

df_genres = df_genres.reindex(cat_data.index)


cat_data.drop('genres',inplace = True, axis = 1)
cat_data = cat_data.merge(df_genres,left_index = True,right_index = True)




cat_data.shape




df_plot_keywords = pd.DataFrame(cat_data['plot_keywords'])
df_plot_keywords = pd.DataFrame(df_plot_keywords.plot_keywords.str.split('|').tolist(),columns = ["plot_keywords_"+str(i) for i in  range(0,5)] )
cat_data.drop('plot_keywords',inplace = True, axis = 1)
df_plot_keywords = df_plot_keywords.reindex(cat_data.index)
cat_data = cat_data.merge(df_plot_keywords,left_index = True,right_index = True)




cat_data.head(2)




cat_data.shape




# cat_data = data[categorical_features]




# # from sklearn.preprocessing import Imputer
# # imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# # imr = imr.fit(cat_data)
# # cat_data = imr.transform(cat_data.values)
# cat_data.fillna(cat_data.mode(),inplace = True)
# cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))





# cat_data = cat_data.loc[:,['color',
#  'Genre_5',
#  'Genre_4',
#  'Genre_3',
#  'content_rating',
#  'Genre_0',
#  'Genre_2',
#  'Genre_1',
#  'language',
#  'country']]

# fig, ax = plt.subplots(figsize=(20,20), ncols=3, nrows=3)
# sns.countplot(data = cat_data, x= 'color', ax = ax[0][0])
# sns.countplot(data = cat_data, x= 'language', ax = ax[0][1])
# sns.countplot(data = cat_data, x= 'country', ax = ax[1][0])
# sns.countplot(data = cat_data, x= 'content_rating', ax = ax[1][1])
# sns.countplot(data = cat_data, x= 'Genre_0', ax = ax[2][0])
# sns.countplot(data = cat_data, x= 'Genre_1', ax = ax[2][1])
# sns.countplot(data = cat_data, x= 'Genre_2', ax = ax[2][2])
# sns.countplot(data = cat_data, x= 'Genre_3', ax = ax[0][2])
# sns.countplot(data = cat_data, x= 'Genre_4', ax = ax[1][2])
# sns.countplot(data = cat_data, x= 'Genre_5', ax = ax[1][1])




cat_data.nunique().sort_values()









cat_data.drop(['movie_imdb_link','Genre_6','Genre_7'],inplace = True, axis = 1)




whole_data = pd.concat([num_data,cat_data],axis = 1)




y = whole_data['imdb_score']




whole_data.drop('imdb_score',axis = 1,inplace = True)




from sklearn.model_selection import train_test_split # to split the data into two parts
X_train,X_test,y_train,y_test = train_test_split(whole_data,y, random_state = 0,test_size = 0.20) # test_size = 0.10




num_feat = whole_data.select_dtypes(exclude=['object']).columns.tolist()
cat_feat = whole_data.select_dtypes(include=['object']).columns.tolist()




X_train_num = X_train[num_feat]

X_train_cat = X_train[cat_feat]




X_test_num = X_test[num_feat]

X_test_cat = X_test[cat_feat]




from scipy.stats import skew 
skewness = X_train_num.apply(lambda x: skew(x.dropna()))
skewness = skewness[abs(skewness) > 0.75]
skew_features = X_train_num[skewness.index]
skew_features  = np.log1p(skew_features)
X_train_num[skewness.index] = skew_features




X_train_num.head()




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)




for i, col in enumerate(num_feat):
    X_train_num.loc[:,col] = X_train_num_scaled[:, i]




from scipy.stats import skew 
skewness = X_test_num.apply(lambda x: skew(x.dropna()))
skewness = skewness[abs(skewness) > 0.75]
skew_features = X_test_num[skewness.index]
skew_features  = np.log1p(skew_features)
X_test_num[skewness.index] = skew_features




X_test_num_scaled = scaler.transform(X_test_num)




for i, col in enumerate(num_feat):
    X_test_num.loc[:,col] = X_test_num_scaled[:, i]




from sklearn.ensemble import RandomForestRegressor
dt = RandomForestRegressor(n_estimators = 1000,n_jobs=-1,random_state = 0)
dt.fit(X_train_num, y_train)
dt_score_train = dt.score(X_train_num, y_train)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(X_test_num_scaled, y_test)
print("Testing score: ",dt_score_test)




df = pd.DataFrame(data = dt.feature_importances_,index = X_train_num.columns.tolist())

df = df[df.iloc[:,0] > 0].sort_values(by = 0,ascending = False)
fig, ax = plt.subplots(figsize=(20,10))
sns.barplot(y = df.index, x= df[0])
plt.xlabel('importance')




sns.set(style="ticks")
sns.pairplot(X_train_num.iloc[:,:4],diag_kind="kde")




train_tar_enc = pd.concat([X_train_cat,y_train],axis = 1)
test_tar_enc = pd.concat([X_test_cat,y_test],axis = 1)




train_tar_enc.head(1)




# # This way we have randomness and are able to reproduce the behaviour within this cell.
# np.random.seed(13)
# from sklearn.model_selection import KFold
# def impact_coding(data, feature, target='imdb_score'):
#     '''
#     In this implementation we get the values and the dictionary as two different steps.
#     This is just because initially we were ignoring the dictionary as a result variable.
    
#     In this implementation the KFolds use shuffling. If you want reproducibility the cv 
#     could be moved to a parameter.
#     '''
#     n_folds = 20
#     n_inner_folds = 10
#     impact_coded = pd.Series()
    
#     oof_default_mean = data[target].mean() # Gobal mean to use by default (you could further tune this)
#     kf = KFold(n_splits=n_folds, shuffle=True)
#     oof_mean_cv = pd.DataFrame()
#     split = 0
#     for infold, oof in kf.split(data[feature]):
#             impact_coded_cv = pd.Series()
#             kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
#             inner_split = 0
#             inner_oof_mean_cv = pd.DataFrame()
#             oof_default_inner_mean = data.iloc[infold][target].mean()
#             for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
#                 # The mean to apply to the inner oof split (a 1/n_folds % based on the rest)
#                 oof_mean = data.iloc[infold_inner].groupby(by=feature)[target].mean()
#                 impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
#                             lambda x: oof_mean[x[feature]]
#                                       if x[feature] in oof_mean.index
#                                       else oof_default_inner_mean
#                             , axis=1))

#                 # Also populate mapping (this has all group -> mean for all inner CV folds)
#                 inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
#                 inner_oof_mean_cv.fillna(value=oof_default_inner_mean, inplace=True)
#                 inner_split += 1

#             # Also populate mapping
#             oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
#             oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
#             split += 1
            
#             impact_coded = impact_coded.append(data.iloc[oof].apply(
#                             lambda x: inner_oof_mean_cv.loc[x[feature]].mean()
#                                       if x[feature] in inner_oof_mean_cv.index
#                                       else oof_default_mean
#                             , axis=1))

#     return impact_coded, oof_mean_cv.mean(axis=1), oof_default_mean

# impact_coding_map = {}
# for f in cat_feat:
#     print("Impact coding for {}".format(f))
#     train_tar_enc["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train_tar_enc, f)
#     impact_coding_map[f] = (impact_coding_mapping, default_coding)
#     mapping, default_mean = impact_coding_map[f]
#     test_tar_enc["impact_encoded_{}".format(f)] = test_tar_enc.apply(lambda x: mapping[x[f]]
#                                                                        if x[f] in mapping
#                                                                          else default_mean
#                                                                , axis=1)




# merged_train.drop(cat_features,inplace = True,axis = 1)
# merged_test.drop(cat_features,inplace = True,axis = 1)

# merged_train.drop('imdb_score',inplace = True,axis = 1)
# merged_test.drop('imdb_score',inplace = True,axis = 1)




import copy
X_train_hash = copy.copy(X_train_cat)
X_test_hash = copy.copy(X_test_cat)
from sklearn.feature_extraction import FeatureHasher
for i in range(X_train_cat.shape[1]):
    X_train_hash.iloc[:,i]=X_train_hash.iloc[:,i].astype('str')
for i in range(X_test_hash.shape[1]):
    X_test_hash.iloc[:,i]=X_test_hash.iloc[:,i].astype('str')
h = FeatureHasher(n_features=10000,input_type="string")




X_train_hash = h.transform(X_train_hash.values)
X_test_hash = h.transform(X_test_hash.values)




X_train_cat.head()




X_train_cat.isnull().sum()




for i in cat_feat:
    print('Feature: ',i)
    print(X_train_cat[i].value_counts()[:7].sum())
    print('--------------------------------------')

    




X_train_cat.drop(['Genre_2','Genre_3','Genre_4','Genre_5'],axis = 1,inplace = True)
X_test_cat.drop(['Genre_2','Genre_3','Genre_4','Genre_5'],axis = 1,inplace = True)




# X_train_cat = X_train_cat.astype('str')




# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder(handle_unknown='ignore')
# enc.fit(X_train_cat)
# X_train_one_hot = enc.transform(X_train_cat)
# X_test_one_hot = enc.transform(X_test_cat)




temp_cat = pd.concat([X_train_cat,X_test_cat])




temp_cat.shape




0.90*3756




temp_cat.country.value_counts()[:10].index.tolist()




# cat_data.loc[cat_data[(cat_data["country"] == "USA")].index,"country"] = 2
# temp_cat.loc[temp_cat[(cat_data["country"] != "USA")&(cat_data["country"] != "UK")&(cat_data["country"] != "Germany")&(cat_data["country"] != "France")].index,"country"] = "Other"
temp_cat.loc[temp_cat[~temp_cat["country"].isin(['USA',
 'UK',
 'France',
 'Germany'])].index,"country"] = "Other"

# cat_data.loc[cat_data[(cat_data["country"] != 2) & (cat_data["country"] != 1)].index,"country"] = 0

temp_cat.country.value_counts()




cat_data.language.value_counts()[:5]




temp_cat["language"] = (temp_cat["language"] == "English") * 1
temp_cat.language.value_counts()




temp_cat.content_rating.value_counts()[:10]

# temp_cat.loc[temp_cat[(temp_cat["content_rating"] == "R")].index,"content_rating"] = 4
# temp_cat.loc[temp_cat[(temp_cat["content_rating"] == "PG-13")].index,"content_rating"] = 3
# temp_cat.loc[temp_cat[(temp_cat["content_rating"] == "PG")].index,"content_rating"] = 2
# temp_cat.loc[temp_cat[(temp_cat["content_rating"] != 4) & (temp_cat["content_rating"] != 3)&(temp_cat["content_rating"] != 2)].index,"content_rating"] = 0




temp_cat.loc[temp_cat[(temp_cat["content_rating"] != "R")&(temp_cat["content_rating"] != "PG-13")&(temp_cat["content_rating"] != "PG")].index,"content_rating"] = "Other"

temp_cat.content_rating.value_counts()




temp_cat.Genre_0.unique()

temp_cat.Genre_0.value_counts()

# temp_cat.loc[temp_cat[(temp_cat["Genre_0"] == "Action")].index,"Genre_0"] = 5
# temp_cat.loc[temp_cat[(temp_cat["Genre_0"] == "Comedy")].index,"Genre_0"] = 4
# temp_cat.loc[temp_cat[(temp_cat["Genre_0"] == "Drama")].index,"Genre_0"] = 3
# temp_cat.loc[temp_cat[(temp_cat["Genre_0"] == "Adventure")].index,"Genre_0"] = 2
# temp_cat.loc[temp_cat[(temp_cat["Genre_0"] != 5) & (temp_cat["Genre_0"] != 4) & (temp_cat["Genre_0"] != 3)&(temp_cat["Genre_0"] != 2)].index,"Genre_0"] = 0




temp_cat.loc[temp_cat[(temp_cat["Genre_0"] != "Action")&(temp_cat["Genre_0"] != "Drama")&(temp_cat["Genre_0"] != "Comedy")&(temp_cat["Genre_0"] != "Adventure")&(temp_cat["Genre_0"] != "Crime")&(temp_cat["Genre_0"] != "Biography")].index,"Genre_0"] = "Other"

temp_cat.Genre_0.value_counts()




temp_cat.Genre_1.value_counts()

# temp_cat["Genre_0"] = ((temp_cat["Genre_0"] == "Comedy") | (temp_cat["Genre_0"] == "Action") |(temp_cat["Genre_0"] == "Drama")|(temp_cat["Genre_0"] == "Adventure") | (temp_cat["Genre_0"] == "Crime") |(temp_cat["Genre_0"] == "Biography")|(temp_cat["Genre_0"] == "Horror")) * 1
# temp_cat.Genre_0.value_counts()


temp_cat.Genre_1.value_counts().index.tolist()

# # temp_cat["Genre_1"] = ((temp_cat["Genre_1"] == "Comedy") | (temp_cat["Genre_1"] == "Action") |(temp_cat["Genre_1"] == "Drama")|(temp_cat["Genre_1"] == "Adventure") | (temp_cat["Genre_1"] == "Crime") |(temp_cat["Genre_1"] == "Romance")|(temp_cat["Genre_1"] == "Mystery")|(temp_cat["Genre_1"] == "Romance")|(temp_cat["Genre_1"] == "Mystery")) * 1
# # temp_cat.Genre_1.value_counts()
# temp_cat.loc[temp_cat[(temp_cat["Genre_1"] == "Drama")].index,"Genre_1"] = 5
# temp_cat.loc[temp_cat[(temp_cat["Genre_1"] == "Adventure")].index,"Genre_1"] = 4
# temp_cat.loc[temp_cat[(temp_cat["Genre_1"] == "Crime")].index,"Genre_1"] = 3
# temp_cat.loc[temp_cat[(temp_cat["Genre_1"] == "Comedy")].index,"Genre_1"] = 2
# temp_cat.loc[temp_cat[(temp_cat["Genre_1"] == "Romance")].index,"Genre_1"] = 1

temp_cat.loc[temp_cat[~temp_cat["Genre_1"].isin(['Drama',
 'Adventure',
 'Crime',
 'Comedy',
 'Romance',
 'Mystery',
 'Thriller',
 'Horror',
 'Family',
 'Animation',
 'Fantasy'])].index,"Genre_1"] = "Other"
temp_cat.Genre_1.value_counts()




# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# pca = PCA(n_components=None, svd_solver="full")
# pca.fit(StandardScaler().fit_transform(num_data))
# # X_train = pca.transform(X_train)
# # pca_data = pca.transform(num_data)
# cum_var_exp = np.cumsum(pca.explained_variance_ratio_)

# temp_cat.drop(['Genre_3','Genre_4','Genre_5'],inplace = True, axis = 1)


temp_cat["color"] = (temp_cat["color"] == "Color") * 1
temp_cat.color.value_counts()

temp_cat.columns.tolist()




temp_cat.drop(['movie_title'],inplace = True, axis = 1)




from sklearn.preprocessing import LabelEncoder
abc  = cat_data[[
 'director_name',
 'actor_2_name',
 'actor_1_name',
 'actor_3_name',
 'plot_keywords_0',
 'plot_keywords_1',
 'plot_keywords_2',
'plot_keywords_3',
 'plot_keywords_4']].astype(str).apply(LabelEncoder().fit_transform)




temp_cat[[
 'director_name',
 'actor_2_name',
 'actor_1_name',
 'actor_3_name',
 'plot_keywords_0',
 'plot_keywords_1',
 'plot_keywords_2','plot_keywords_3',
 'plot_keywords_4']] = abc




temp_cat = pd.get_dummies(temp_cat)




temp_cat.head()




X_train_cat = temp_cat.loc[X_train_cat.index,:]




X_test_cat = temp_cat.loc[X_test_cat.index,:]




X_train = pd.concat([X_train_num,X_train_cat], axis =1)




X_test = pd.concat([X_test_num,X_test_cat], axis =1)




from sklearn.ensemble import RandomForestRegressor
dt = RandomForestRegressor(n_estimators = 1000,n_jobs=-1,random_state = 0)
dt.fit(X_train, y_train)
dt_score_train = dt.score(X_train, y_train)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(X_test, y_test)
print("Testing score: ",dt_score_test)




from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridge.fit(X_train,y_train)
alpha = ridge.alpha_
print('best alpha',alpha)
print("Try again for more precision with alphas centered around " + str(alpha))
ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv = 5)
ridge.fit(X_train, y_train)
alpha = ridge.alpha_
print("Best alpha :", alpha)
# print("Ridge RMSE on Training set :", rmse_CV_train(ridge).mean())
# print("Ridge RMSE on Test set :", rmse_CV_test(ridge).mean())
y_train_rdg = ridge.predict(X_train)
y_test_rdg = ridge.predict(X_test)
print("Training score: ",ridge.score(X_train,y_train))
print("Testing score: ",ridge.score(X_test,y_test))




plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue",  label = "Training data")
#plt.scatter(y_test_rdg,y_test_rdg - y_test, c = "green",  label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()





from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)




from sklearn.linear_model import Ridge
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]




cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")




cv_ridge




linridge = Ridge(alpha=5).fit(X_train, y_train)




linridge.score(X_train, y_train)

print('IMDB dataset')

print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
# print('ridge regression features: {}'
#      .format(features))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, y_train)))
# print('R-squared score (test): {:.3f}'
#      .format(linridge.score(X_test, y_test)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))
print('Number of zero features: {}'
     .format(np.sum(linridge.coef_ == 0)))




from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


#lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 3
lasso_cv = LassoCV(alphas=alphas, random_state=0)
# lasso_cv = Lasso(alpha = 0.001)
lasso_cv.fit(X_train, y_train)
#lasso_cv.predict(X_test)
print("Training score: ",lasso_cv.score(X_train, y_train))
print("Testing score: ",lasso_cv.score(X_test, y_test))





tuned_parameters = [{'alpha': alphas}]
n_folds = 3
ridge_cv = RidgeCV(alphas=alphas)
ridge_cv.fit(X_train, y_train)
print("Training score: ",ridge_cv.score(X_train, y_train))
print("Testing score: ",ridge_cv.score(X_test, y_test))




# from sklearn import neighbors
# def func(distances):
#     w = []
#     kek = 0.0
#     for dist in distances:
#         kek += np.exp(dist)
#     for dist in distances:
#         w.append(np.exp(dist)/kek)
#     return w
# knn = neighbors.KNeighborsRegressor(n_neighbors = 3, weights = func)
# knn = knn.fit(X_train, y_train)
# print("Training score: ",knn.score(X_train, y_train))
# print("Testing score: ",knn.score(X_test, y_test))




# from sklearn.decomposition import PCA
# pca = PCA(n_components = 90).fit(X_train)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# # plt.xlim(0,7,1)
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')




# np.cumsum(pca.explained_variance_ratio_)




# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)




# plt.scatter(y_test, y_test_ - y_test, c = "blue",  label = "Training data")
# #plt.scatter(y_test_rdg,y_test_rdg - y_test, c = "green",  label = "Validation data")
# plt.title("Linear regression")
# plt.xlabel("Predicted values")
# plt.ylabel("Residuals")
# plt.legend(loc = "upper left")
# plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
# plt.show()




(data['imdb_score']).min()




temp_whole = pd.concat([X_train,X_test])




temp_whole.shape




target = pd.concat([y_train,y_test])




target_classes = pd.cut(target,bins = [0,6,10],labels = [0,1],right = True,include_lowest = True)




y.size




target_classes.value_counts()




from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
pca = PCA(n_components=9, svd_solver="full")
pca_data = pca.fit_transform(temp_whole)
# X_train = pca.transform(X_train)
# pca_data = pca.transform(num_data)
cum_var_exp = np.cumsum(pca.explained_variance_ratio_)




cum_var_exp




target_classes.isnull().any()




from sklearn.model_selection import train_test_split # to split the data into two parts
X_train,X_test,y_train,y_test = train_test_split(temp_whole,target_classes, random_state = 1,test_size = 0.20,stratify =target_classes) # test_size = 0.10




from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train,y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))




from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier(n_estimators = 1000,n_jobs=-1,random_state = 0)
dt.fit(X_train, y_train)
dt_score_train = dt.score(X_train, y_train)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(X_test, y_test)
print("Testing score: ",dt_score_test)




df = pd.DataFrame(data = dt.feature_importances_[:10],index = temp_whole.columns.tolist()[:10])

df = df[df.iloc[:,0] > 0].sort_values(by = 0,ascending = False)
fig, ax = plt.subplots(figsize=(20,10))
sns.barplot(y = df.index, x= df[0])
plt.xlabel('importance')




from sklearn.learning_curve import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
LogisticRegression(penalty='l2', random_state=0),
X=X_train,
y=y_train,
param_name='C',
param_range=param_range,cv=10)




train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean,
color='blue', marker='o',
markersize=5,
label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
train_mean - train_std, alpha=0.15,
color='blue')
plt.plot(param_range, test_mean,
color='green', linestyle='--',
marker='s', markersize=5,
label='validation accuracy')
plt.fill_between(param_range,
test_mean + test_std,
test_mean - test_std,
alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.7, 0.85])
plt.show() 




from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C = 0.1,penalty='l2', random_state=0)

clf.fit(X_train,y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))




# >>> from sklearn.grid_search import GridSearchCV
# >>> from sklearn.pipeline import Pipeline
# >>> from sklearn.svm import SVC
# >>> pipe_svc = Pipeline([('clf', SVC(random_state=1))])
# >>> param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# >>> param_grid = [{'clf__C': param_range,
# ... 'clf__kernel': ['linear']},
# ... {'clf__C': param_range,
# ... 'clf__gamma': param_range,
# ... 'clf__kernel': ['rbf']}]
# >>> gs = GridSearchCV(estimator=pipe_svc,
# ... param_grid=param_grid,
# ... scoring='accuracy',
# ... cv=10,
# ... n_jobs=-1)
# >>> gs = gs.fit(X_train, y_train)


# print(gs.best_score_)
# print(gs.best_params_)




from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(cm)




num_data.columns






