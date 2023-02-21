#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




# 패키지 불러오기
import pandas as pd # 분석
import matplotlib.pyplot as plt # 시각화
import seaborn as sns # 시각화
import numpy as np # 분석
from scipy.stats import norm # 분석
from sklearn.preprocessing import StandardScaler # 분석
from scipy import stats # 분석
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import gc




train = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')
test = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')
print("train.csv Shape : ", train.shape)
print("test.csv Shape : ", test.shape)




train.head(2)




train['price'].describe()




# histogram
f, ax = plt.subplots(figsize = (8,6))
sns.distplot(train['price'])




#skewness and kurtosis (왜도와 첨도)
print("Skewness : %f " % train['price'].skew())
print("Kurtosis : %f " % train['price'].kurt())




fig = plt.figure(figsize = (15,10))

fig.add_subplot(1,2,1)
res = stats.probplot(train['price'], plot = plt)

fig.add_subplot(1,2,2)
res = stats.probplot(np.log1p(train['price']), plot = plt)




train['price'] = np.log1p(train['price'])
#histogram
f, ax = plt.subplots(figsize = (8, 6))
sns.distplot(train['price'])




#saleprice correlation matrix
k = 10 # 히트맵 변수 수
corrmat = abs(train.corr(method = "spearman"))
cols = corrmat.nlargest(k, 'price').index # nlargest : Return this many descending sorted values
cm = np.corrcoef(train[cols].values.T) # correlation 특정 컬럼에 대해서
sns.set(font_scale = 1.25)
f, ax = plt.subplots(figsize = (18,8))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True,
                fmt = '.2f',annot_kws = {'size' : 8}, yticklabels = cols.values,
                xticklabels = cols.values)
plt.show()




# 상관계수가 가장 낮음 10개 반응변수 
cols = corrmat.nsmallest(k, 'price').index # nsmallest : Return this many descending sorted values
cm = np.corrcoef(train[cols].values.T) # correlation 특정 컬럼에 대해서
sns.set(font_scale = 1.25)
f, ax = plt.subplots(figsize = (18,8))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True,
                fmt = '.2f',annot_kws = {'size' : 8}, yticklabels = cols.values,
                xticklabels = cols.values)
plt.show()




data = pd.concat([train['price'], train['grade']], axis = 1)
f, ax = plt.subplots(figsize = (8,6))
fig = sns.boxplot(x = 'grade', y = 'price', data = data)




data = pd.concat([train['price'], train['sqft_living']], axis = 1)
f, ax = plt.subplots(figsize = (8,6))
fig = sns.regplot(x = 'sqft_living', y = 'price', data = data)




data = pd.concat([train['price'], train['sqft_living15']], axis = 1)
f, ax = plt.subplots(figsize = (8,6))
fig = sns.regplot(x = 'sqft_living15', y = 'price', data = data)




data = pd.concat([train['price'], train['sqft_above']], axis = 1)
f, ax = plt.subplots(figsize = (8,6))
fig = sns.regplot(x = 'sqft_above', y = 'price', data = data)




data = pd.concat([train['price'], train['lat']], axis = 1)
f, ax = plt.subplots(figsize = (8,6))
fig = sns.regplot(x = 'lat', y = 'price', data = data)




train_latlong = pd.concat([train['price'], train['long'], train['lat']], axis = 1)
train_latlong['price'] = np.log1p(train_latlong['price'])
train_latlong['long'] = np.log1p(abs(train_latlong['long']))
train_latlong['lat'] = np.log1p(train_latlong['lat'])




f, ax = plt.subplots(figsize = (8,6))
fig = sns.regplot(x = 'lat', y = 'price', data = train_latlong)




f, ax = plt.subplots(figsize = (8,6))
fig = sns.regplot(x = 'long', y = 'price', data = train_latlong)




data = pd.concat([train['price'], train['long']], axis = 1)
f, ax = plt.subplots(figsize = (8,6))
fig = sns.regplot(x = 'long', y = 'price', data = data)




test.head()




data = pd.concat([train['price'], train['bathrooms']], axis = 1)
f, ax = plt.subplots(figsize = (18,6))
fig = sns.boxplot(x = 'bathrooms', y = 'price', data = data)




data = pd.concat([train['price'], train['bedrooms']], axis = 1)
f, ax = plt.subplots(figsize = (18,6))
fig = sns.boxplot(x = "bedrooms", y= "price", data =data)




train.isnull().sum()




test.isnull().sum()




def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color




import plotly.graph_objs as go
import random
import plotly.offline as py
from plotly import tools
py.init_notebook_mode(connected=True)
from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import time

train_unique = []
columns = ['bedrooms', 'bathrooms','floors','waterfront','view',
          'condition','grade']

for i in columns :
    train_unique.append(len(train[i].unique()))

unique_train = pd.DataFrame()
unique_train['Columns'] = columns
unique_train['Unique_value'] = train_unique

data = [
    go.Bar(
    x = unique_train['Columns'],
    y = unique_train['Unique_value'],
    name = 'Unique value in features',
    textfont = dict(size = 20),
    marker = dict(
    line = dict(
    color = generate_color(),
    #width = 2,
    ), opacity = 0.45
    )
    ),
]

layout = go.Layout(title = " Unique Value By Column",
                  xaxis = dict(title = 'Columns', ticklen = 5,
                              zeroline = False, gridwidth = 2),
                  yaxis = dict(title = 'Value Count',
                              ticklen = 5, gridwidth = 2),
                  showlegend = True)
fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = 'skin')




data = pd.concat([train['price'], train['sqft_living']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.regplot(x='sqft_living', y="price", data=data)




train.loc[train['sqft_living'] > 13000]




test.loc[test['sqft_living'] > 13000]




train = train.loc[train['id'] != 8912]




data = pd.concat([train['price'], train['grade']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='grade', y="price", data=data)




train.loc[(train['price'] > 14.7) & (train['grade'] == 8)]




train.loc[(train['price'] > 15.5) & (train['grade'] == 11)]




train = train.loc[train['id'] != 7173]
train = train.loc[train['id'] != 2775]




data = pd.concat([train['price'], train['bedrooms']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='bedrooms', y="price", data=data)




train.loc[train['bedrooms'] >= 10]




## 테스트도 확인
test.loc[test['bedrooms'] >= 10]




train_latlong = pd.concat([train['price'], train['long'], train['lat']], axis = 1)
train_latlong['price'] = np.log1p(train_latlong['price'])
train_latlong['long'] = np.log1p(abs(train_latlong['long']))




f, ax = plt.subplots(figsize = (8,6))
fig = sns.regplot(x = 'long', y = 'price', data = train_latlong)




train.loc[np.log1p(abs(train['long'])) < 4.807]




test.loc[np.log1p(abs(test['long'])) < 4.807]




train = train.loc[train['long'] != -121.359]
train = train.loc[train['long'] != -121.315]
train = train.loc[train['long'] != -121.352]
train = train.loc[train['long'] != -121.319]
train = train.loc[train['long'] != -121.316]
train = train.loc[train['long'] != -121.321]
train = train.loc[train['long'] != -121.325]




for df in [train, test] :
    df['date'] = df['date'].apply(lambda x : x[0:8])
    df['yr_renovated'] = df['yr_renovated'].apply(lambda x : np.nan if x == 0 else x)
    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])
    df['month'] = df['date'].apply(lambda x : x[4:6])
    df['year'] = df['date'].apply(lambda x : x[0:4])
    df['day'] = df['date'].apply(lambda x : x[6:8])




del train['date']
del test['date']




train.info()




for df in [train, test]:
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['grade_condition'] = df['grade'] * df['condition']
    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']
    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']
    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 
    df['is_renovated'] = df['yr_renovated'] - df['yr_built']
    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)
    df['abslong'] = abs(df['long'])
    df['month'] = df['month'].astype('int')
    df['floor_grade'] = df['floors'] * df['grade']
    df['floor_view'] = df['floors'] + df['view']
    df['grade_bed'] = df['grade'] * df['bathrooms']
    df['long_lat'] = df['long'] + df['lat']
    df['longlat'] = df['long'] * df['lat']
    df['view_water'] = df['view'] + df['waterfront']
    df['long_bathroom'] = df['long'] * df['bathrooms']
    df['year'] = df['year'].astype('int')
    df['day'] = df['day'].astype('int')




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import RidgeCV

y_reg = train['price']
del train['price']
del train['id']
test_id = test['id']
del test['id']

kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, train, y_reg, 
                                   scoring="neg_mean_squared_error", 
                                   cv = kfolds))
    return(rmse)

def ridge_selector(k):
    ridge_model = make_pipeline(RobustScaler(),
                                RidgeCV(alphas = [k],
                                        cv=kfolds)).fit(train, y_reg)
    
    ridge_rmse = cv_rmse(ridge_model).mean()
    return(ridge_rmse)

r_alphas = [.0001, .0003, .0005, .0007, .0009, 
          .01, 0.05, 0.1, 0.3, 1, 3, 5,6,7,8,9,10]

ridge_scores = []
for alpha in r_alphas:
    score = ridge_selector(alpha)
    ridge_scores.append(score)
    
plt.plot(r_alphas, ridge_scores, label='Ridge')
plt.legend('center')
plt.xlabel('alpha')
plt.ylabel('score')




alphas_alt = [5.8,5.9,6,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,7]

ridge_model2 = make_pipeline(RobustScaler(),
                            RidgeCV(alphas = alphas_alt,
                                    cv=kfolds)).fit(train, y_reg)

print("Ridge rmse : ",cv_rmse(ridge_model2).mean())




print("Best of alpha in ridge model :" ,ridge_model2.steps[1][1].alpha_)





ridge_coef = pd.DataFrame(np.round_(ridge_model2.steps[1][1].coef_, decimals=3), 
test.columns, columns = ["penalized_regression_coefficients"])
# remove the non-zero coefficients
ridge_coef = ridge_coef[ridge_coef['penalized_regression_coefficients'] != 0]
# sort the values from high to low
ridge_coef = ridge_coef.sort_values(by = 'penalized_regression_coefficients', 
ascending = False)

# plot the sorted dataframe
fig = plt.figure(figsize = (25,25))
ax = sns.barplot(x = 'penalized_regression_coefficients', y= ridge_coef.index , 
data=ridge_coef)
ax.set(xlabel='Penalized Regression Coefficients')




## lightgbm
train_columns = [c for c in train.columns if c not in ['id']]




import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import RidgeCV



param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 4950}

#prepare fit model with cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
feature_importance_df = pd.DataFrame()

#run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx][train_columns], label=y_reg.iloc[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][train_columns],
                           label=y_reg.iloc[val_idx])#, categorical_feature=categorical_feats)
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], 
                    verbose_eval=500, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(train.iloc[val_idx][train_columns], 
                               num_iteration=clf.best_iteration)
    #feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, 
                                       fold_importance_df], axis=0)
    #predictions
    predictions += clf.predict(test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits
    
cv = np.sqrt(mean_squared_error(oof, y_reg))
print(cv)




##plot the feature importance
cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')




## xgboost




import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import model_selection
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns




train = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/train.csv')
test = pd.read_csv('../input/2019-2nd-ml-month-with-kakr/test.csv')




y_train = train.price
x_train = train.drop(['id', 'price'], axis=1)
x_test = test.drop(['id'], axis=1)
for df in [x_train,x_test] :
    df['year'] = df.date.apply(lambda x: x[0:4]).astype(int)
    df['month'] = df.date.apply(lambda x: x[4:6]).astype(int)
    df['day'] = df.date.apply(lambda x: x[6:8]).astype(int)




for df in [x_train, x_test] :
    df['year_month'] = df['year']*100 + df['month']
    df['month_day'] = df['month']*100 + df['day']
    df['ym_freq'] = df.groupby('year_month')['year_month'].transform('count')
    df['md_freq'] = df.groupby('month_day')['month_day'].transform('count')




def preprocessing(df):
    # log
    df.sqft_living2 = np.log(df.sqft_living)
    df.sqft_lot2 = np.log(df.sqft_lot)
    df.sqft_above2 = np.log(df.sqft_above)
    df.sqft_basement2 = np.log(df.sqft_basement)
    df.sqft_lot152 = np.log(df.sqft_lot15)
    
    df['roomsum'] = np.log(df.bedrooms + df.bathrooms)
    df['roomsize'] = df.sqft_living / df.roomsum
    
    df['pos'] = df.long.astype(str) + ', ' + df.lat.astype(str)
    df['density'] = df.groupby('pos')['pos'].transform('count')
    
    df = df.drop(['pos'], axis=1)
    
    return df

x_train = preprocessing(x_train)
x_test = preprocessing(x_test)




for df in [x_train, x_test]:
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['grade_condition'] = df['grade'] * df['condition']
    df['sqft_total'] = df['sqft_living'] + df['sqft_lot']
    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']
    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 
    df['is_renovated'] = df['yr_renovated'] - df['yr_built']
    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)
    df['abslong'] = abs(df['long'])
    df['month'] = df['month'].astype('int')
    df['floor_grade'] = df['floors'] * df['grade']
    df['floor_view'] = df['floors'] + df['view']
    df['grade_bed'] = df['grade'] * df['bathrooms']
    df['long_lat'] = df['long'] + df['lat']
    df['longlat'] = df['long'] * df['lat']
    df['view_water'] = df['view'] + df['waterfront']
    df['long_bathroom'] = df['long'] * df['bathrooms']
    df['year'] = df['year'].astype('int')
    df['day'] = df['day'].astype('int')
    

del x_train['date']
del x_test['date']




xgb_params = {
    'eta': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

print('Transform DMatrix...')
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

print('Start Cross Validation...')

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=20,verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
print('best num_boost_rounds = ', len(cv_output))
rounds = len(cv_output)




model = xgb.train(xgb_params, dtrain, num_boost_round = rounds)
preds = model.predict(dtest)





sub = test[['id']]
sub['price'] = preds
sub.to_csv('goodnight.csv', index=False)




test_ridge_preds = np.expm1(ridge_model2.predict(test))




test_ensemble_preds = 0.7*test_lgb_preds + 0.3*test_ridge_preds





test_lgb_preds = np.expm1(predictions)




submission0 = pd.DataFrame({'id': test_id, 'price': test_ridge_preds})
submission0.to_csv('ridge4_.csv', index=False)




submission = pd.DataFrame({'id': test_id, 'price': test_lgb_preds})
submission.to_csv('lightgbm12_rate001.csv', index=False)




test_xgb_preds = np.expm1(y_pred)
submission2 = pd.DataFrame({'id' : test_id, 'price' : y_pred})
submission2.to_csv('xgboost.csv', index = False)




submission1 = pd.DataFrame({'id': test_id, 'price': test_ensemble_preds})
submission1.to_csv('ensemble.csv', index=False)




gg = pd.read_csv('gg.csv')
xgb = pd.read_csv('sub_xgb3.csv')




library(xgboost)
library(ranger)
library(tictoc)
library(corrplot)
library(tidyverse)
library(gridExtra)
library(data.table)
library(caret)




train <- fread("../input/2019-2nd-ml-month-with-kakr/train.csv")
test <- fread("../input/2019-2nd-ml-month-with-kakr/test.csv")




train[ , filter:="train"] # train/test를 구분할 key변수 "filter"를 만듭니다.
test[ , ":="(price=-999, filter="test")] # train/test를 구분할 key변수 "filter"를 만들고, row bind를 위해 test set에 price 변수를 만들고 모든 값을 -999로 채우겠습니다.
full <- rbind(train, test)
full[,filter:=factor(filter, levels=c("train", "test"))]




## transform to factor type for categorical variables
full[, yyyymm:=factor(yyyymm)]
full[, zipcode:=factor(zipcode)]

cat_vars <- c("waterfront", "yyyymm", "zipcode")
del_vars <- c("id", "filter", "price", "mm", "dd", "yyyy", "yr_built", "date", "sqft_basement")
num_vars <- setdiff(colnames(full), c(cat_vars, del_vars))

## 수치형 변수 표준화
X_train_num <- full[filter=="train",num_vars, with=F]
X_test_num <- full[filter=="test",num_vars, with=F]

mean.tr <- apply(X_train_num, 2, mean)
sd.tr <- apply(X_train_num, 2, sd)

X_train_num <- scale(X_train_num, center=mean.tr, scale=sd.tr)
X_test_num <- scale(X_test_num, center=mean.tr, scale=sd.tr)

X_train <- model.matrix(~.-1, data=cbind(X_train_num, full[filter=="train", cat_vars, with=F])) 
X_test <- model.matrix(~.-1, data=cbind(X_test_num, full[filter=="test", cat_vars, with=F]))
Y_train <- log(full[filter=="train", price])




tuneGrid <- expand.grid(
  max_depth             = c(6, 60),        # default : 6
  subsample             = c(0.8, 1),       # default : 1
  colsample_bytree      = c(0.9, 1)        # default : 1
)

RMSE_exp <- function(preds, dtrain) {
  labels <- xgboost::getinfo(dtrain, "label")
  err <- sqrt(mean((exp(labels)-exp(preds))^2))
  return(list(metric = "RMSE_exp", value = err))
}

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label= Y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test))
results <- list(val_rmse = rep(0, nrow(tuneGrid)),
                nrounds = rep(0, nrow(tuneGrid)))
for (i in 1:nrow(tuneGrid)){
    params <- list(
      objective         = "reg:linear",
      metric            = "rmse",
      booster           = "gbtree",
      eta               = 0.01,                         # default : 0.3
      gamma             = 0,                            # default : 0
      min_child_weight  = 1,                            # default : 1
      max_depth         = tuneGrid[i, "max_depth"],
      subsample         = tuneGrid[i, "subsample"],
      colsample_bytree  = tuneGrid[i, "colsample_bytree"]
    )
   # tic("xgbcv")
    xgbcv <- xgb.cv(params = params, 
                    data = dtrain, 
                    nfold = 5, 
                    nrounds = 10000,
                    early_stopping_rounds = 100,
                    feval = RMSE_exp,
                    print_every_n = 100,
                    maximize = F, 
                    seed=42)
   # toc()
    results[["val_rmse"]][i] <- unlist(xgbcv$evaluation_log[xgbcv$best_iteration, "test_RMSE_exp_mean"])
    results[["nrounds"]][i] <- xgbcv$best_iteration
}




min.index <- which.min(results[["val_rmse"]])
tuneGrid[min.index, ]




cbind(tuneGrid, RMSE=unlist(results[["val_rmse"]]))




default_param <- list(
      objective         = "reg:linear",
      booster           = "gbtree",
      eta               = 0.01,
      gamma             = 0,
      min_child_weight  = 1,
      max_depth         = tuneGrid[min.index, "max_depth"],
      subsample         = tuneGrid[min.index, "subsample"],
      colsample_bytree  = tuneGrid[min.index, "colsample_bytree"]
    )

# train the model using the best iteration found by cross validation
fit.xgb <- xgb.train(data = dtrain, 
                     params = default_param, 
                     nrounds = results[["nrounds"]][min.index], 
                     seed=42)




predictions_xgb <- exp(predict(fit.xgb, dtest)) # need to reverse the log to the real values
head(predictions_xgb)




submission_xgb <- read.csv('../input/sample_submission.csv')
submission_xgb$price <- predictions_xgb
write.csv(submission_xgb, file='submission_xgb.csv', row.names = F)




submi = pd.read_csv('../input/submission3/submission123.csv')
goodnight = pd.read_csv('../input/submission3/goodnight.csv')




## 앙상블
k = (submi + goodnight)/2
k.to_csv('dataal.csv', index = False)

