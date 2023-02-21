import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn

from scipy.stats import skew
from scipy.stats.stats import pearsonr

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Training Data Shape: ",train.shape)
print("Testing Data Shape: ", test.shape)
print(train.head())

print(test.head())

prices = pd.DataFrame({"price":train["SalePrice"], \
                       "log(price + 1)":np.log1p(train["SalePrice"]), \
                       "log(price)":np.log(train["SalePrice"]) \
                      })
_ = prices.hist()

all_train = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
print(all_train.shape)

# Log the output variable
train['SalePrice'] = np.log1p(train['SalePrice'])

# Find all skewed features that is > 0.75 and log them
numeric_features = all_train.dtypes[all_train.dtypes != "object"].index
skewed_features = all_train[numeric_features].apply(lambda x: skew(x.dropna()))
skewed_features = skewed_features[skewed_features > 0.75].index
all_train[skewed_features] = np.log1p(all_train[skewed_features])
print(all_train[skewed_features].shape)

# Convert categorical features using dummies
all_train = pd.get_dummies(all_train)
print(all_train.dtypes.index)

# Fill in missing values with mean
all_train = all_train.fillna(all_train.mean())
print(all_train.head())

# Replace infinity values with 0
all_train = all_train.replace([np.inf, -np.inf], 0)
print(all_train.head())

# Split train/test set back to the way it was.
x_train = all_train[:train.shape[0]]
x_test = all_train[train.shape[0]:]
print(x_train.head())
print(x_test.shape)
y_train = train['SalePrice']
print(y_train.head())

from sklearn.linear_model import Ridge, LassoCV
from sklearn.model_selection import cross_val_score, GridSearchCV

ridge = Ridge(alpha=10)
ridge_rmse = np.sqrt(-cross_val_score(ridge, x_train, y_train, cv=5, scoring='neg_mean_squared_error'))
print(ridge_rmse.mean())

parameters = {'alpha':[0.05, 0.1, 0.3, 1, 3, 5, 7, 9, 10, 15, 30, 50, 75]}
ridge = Ridge()
clf = GridSearchCV(ridge, parameters, cv=10, scoring='neg_mean_squared_error')
clf.fit(x_train, y_train)
score = clf.cv_results_['mean_test_score']
best_param = clf.best_params_
print(score)
print("Best Parameter is: ", best_param)

# Make a plot for RMSE vs Alphas to see where is the minimum

rmse = []
for i in score:
    rmse.append(np.sqrt(-i))
print(rmse)

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 7, 9, 10, 15, 30, 50, 75]
cv_ridge = pd.Series(rmse, index = alphas)
_ = cv_ridge.plot(title = "RMSE vs Alpha Sweep")
plt.xlabel("Alpha")
plt.ylabel("RMSE")

print("Minimum RMSE for Ridge: ", cv_ridge.min())

# Do CV for Lasso
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], cv=10).fit(x_train, y_train)
lasso_score = np.sqrt(-cross_val_score(model_lasso, x_train, y_train, cv=10, scoring="neg_mean_squared_error"))
print(lasso_score.mean())

# Visualize the features selected by Lasso. Top 30.
coef = pd.Series(model_lasso.coef_, index = x_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " \
      +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values(ascending=False).head(30)])
_ = imp_coef.plot(kind = "barh").invert_yaxis()
_ = plt.title("Coefficients in the Lasso Model")

import xgboost as xgb
from xgboost import XGBRegressor

#model = xgb.cv(params={"max_depth":2, "eta":0.1}, \
#               dtrain=xgb.DMatrix(x_train, y_train), \
#               num_boost_round=500, \
#               nfold=5, \
#               early_stopping_rounds=50)
#print(model)
#model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

# Use CV to tune n_estimators needed.
xgb_cv_params = {'max_depth': 4, \
                 'subsample': 0.8, \
                 'n_estimators': 1000, \
                 'learning_rate': 0.1, \
                 'colsample_bytree': 0.5, \
                 'min_child_weight': 2, \
                 'gamma': 0, \
                 'scale_pos_weight': 1, \
                 'seed': 42, \
                }
model = xgb.cv(params=xgb_cv_params, \
               dtrain=xgb.DMatrix(x_train, y_train), \
               num_boost_round=1000, \
               nfold=5, \
               metrics='rmse', \
               early_stopping_rounds=100)

print(model.shape[0]) # This tells us n_estimators is 395
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
print(model)

# Grid Search with XGBoost. Tuning max_depth & min_child_weight
xgb_model = XGBRegressor()
xgb_params = {'max_depth': [6], \
              'subsample': [0.8], \
              'n_estimators': [395], \
              'learning_rate': [0.1], \
              'colsample_bytree': [0.5], \
              'min_child_weight': [1, 2, 3], \
              'gamma': [0], \
              'scale_pos_weight': [1], \
              'seed': [42]
             }
# colsample_bytree = sqrt(288) = 17, because 288 features
xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, verbose=5, scoring='neg_mean_squared_error')
xgb_grid.fit(x_train, y_train)
score = xgb_grid.cv_results_
print("Score: ", score)

mean_test_score = score['mean_test_score']
print("mean_test_score: ", mean_test_score)

mean_test_score = score['mean_test_score'].mean()
print("Average mean_test_score: ", mean_test_score)

#print(xgb_grid.best_score_, np.sqrt(-xgb_grid.best_score_))
#print(xgb_grid.best_params_)
df = pd.DataFrame(score['params'])
df['mean_train_score'] = score['mean_train_score']
df['mean_test_score'] = score['mean_test_score']
print(df)
df.to_csv('xgb_cv_1.csv',header=True, index_label='id')

# Copied Code
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        # TODO: cvresult tells you how many n_estimators you need
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
    
    
    
        alg.set_params(n_estimators=cvresult.shape[0])
    
    
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    score_list = [0.12359611644384301, 0.13679546776117987, 0.12479983974348685, \
              0.14904697246170417, 0.12720849028268513, 0.15097019573412496, \
              0.1275264678409937, 0.13773525329413672, 0.12317061337835418, \
              0.14416657032752081, 0.12711412195346353, 0.15463505424062166, \
              0.13500370365289982, 0.15412008305214475, 0.13478130434151467, \
              0.14624294854795564, 0.14301398533010679, 0.1459554726620417]
print(sorted(score_list))