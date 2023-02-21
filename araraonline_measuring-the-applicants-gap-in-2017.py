#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import RepeatedKFold

np.random.seed(1)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')




get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")




# import data
df = pd.read_pickle('../input/nycschools2017/schools2017.pkl')
print(df.shape[0], "schools")

# drop schools with missing test data
df = df[df.loc[:, 'Mean Scale Score - ELA':'% Level 4 - Math'].notnull().all(axis=1)]
print(df.shape[0], "schools after dropping missing test data")

# drop schools with missing attendance data
df = df[df['Percent of Students Chronically Absent'].notnull()]
print(df.shape[0], "schools after dropping missing attendance data")

# schools with 0-5 SHSAT testers have this value set to NaN
applicantsok = df['# SHSAT Testers'].notnull()

# show head of data
f2_columns = ['Latitude', 'Longitude', 'Economic Need Index',
              'Mean Scale Score - ELA', 'Mean Scale Score - Math']
pct_columns = [c for c in df.columns if c.startswith('Percent')]
pct_columns += [c for c in df.columns if c.startswith('%')]
df.head().style.     format('{:.2f}', subset=f2_columns).     format('{:.1%}', subset=pct_columns)




# data
in_columns = [
    'Charter School?',
    'Percent Asian',
    'Percent Black',
    'Percent Hispanic',
    'Percent Other',
    'Percent English Language Learners',
    'Percent Students with Disabilities',
    'Economic Need Index',
    'Percent of Students Chronically Absent',
    
    'Mean Scale Score - ELA',
    '% Level 2 - ELA',
    '% Level 3 - ELA',
    '% Level 4 - ELA',
    'Mean Scale Score - Math',
    '% Level 2 - Math',
    '% Level 3 - Math',
    '% Level 4 - Math', 
]
inputs = df[applicantsok][in_columns]
outputs = logit(df[applicantsok]['% SHSAT Testers'])  # the logit will be explained later


# cross-validation
cv_results = []
n_splits = 10
n_repeats = 20
for n_components in range(1, inputs.shape[1] + 1):
    mae_scores = []
    mse_scores = []
    
    x = PCA(n_components).fit_transform(inputs)
    x = pd.DataFrame(x, index=inputs.index, columns=["PC{}".format(i) for i in range(1, n_components + 1)])
    x['Constant'] = 1
    y = outputs.copy()
    

    cv = RepeatedKFold(n_splits, n_repeats, random_state=1)    
    for train, test in cv.split(x):
        x_train = x.iloc[train]
        x_test = x.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        
        model = sm.RLM(y_train, x_train, M=sm.robust.norms.HuberT())
        results = model.fit()
        predictions = model.predict(results.params, exog=x_test)
        mae = median_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae_scores.append(mae)
        mse_scores.append(mse)
        
    mae_scores = np.array(mae_scores).reshape(n_repeats, n_splits).mean(axis=1)  # mean of each repeat
    mse_scores = np.array(mse_scores).reshape(n_repeats, n_splits).mean(axis=1)  # mean of each repeat
        
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)
    
    cv_result = (n_components, mae_mean, mse_mean, mae_std, mse_std)
    cv_results.append(cv_result)
    
df_columns = ['n_components', 'mae__mean', 'mse__mean', 'mae__std', 'mse__std']
cv_results_df = pd.DataFrame(cv_results, columns=df_columns)
cv_results_df




# visualize results

cvdf = cv_results_df  # code sugar

plt.figure()
plt.errorbar(cvdf.n_components, cvdf.mae__mean, cvdf.mae__std, marker='o', label='Median Absolute Error')
plt.legend()

plt.figure()
plt.errorbar(cvdf.n_components, cvdf.mse__mean, cvdf.mse__std, marker='o', label='Mean Squared Error')
plt.legend();




base_df = df[[  # explanatory variables
    'Charter School?',
    'Percent Asian',
    'Percent Black',
    'Percent Hispanic',
    'Percent Other',
    'Percent English Language Learners',
    'Percent Students with Disabilities',
    'Economic Need Index',
    'Percent of Students Chronically Absent',
    
    'Mean Scale Score - ELA',
    '% Level 2 - ELA',
    '% Level 3 - ELA',
    '% Level 4 - ELA',
    'Mean Scale Score - Math',
    '% Level 2 - Math',
    '% Level 3 - Math',
    '% Level 4 - Math',
]]

# transform the variables (apply the PCA)
n_components = 8
pca = PCA(n_components)
transformed = pca.fit_transform(base_df)
transformed = pd.DataFrame(transformed, index=base_df.index, columns=["PC{}".format(i+1) for i in range(n_components)])

# add a constant column (needed for our model with statsmodels)
inputs = transformed
inputs.insert(0, 'Constant', 1.0)
inputs.head()




# prepare inputs and outputs
inputs_fit = inputs[applicantsok]
outputs_fit = logit(df['% SHSAT Testers'][applicantsok])
inputs_predict = inputs

# fit the model
model = sm.RLM(outputs_fit, inputs_fit, M=sm.robust.norms.HuberT())
results = model.fit()

# make predictions
predictions = model.predict(results.params, exog=inputs_predict)
predictions = pd.Series(predictions, index=inputs_predict.index)
predictions = expit(predictions)  # expit is the inverse of the logit
predictions.name = 'Expected % of SHSAT Testers'




results.summary()




_predictions = logit(predictions[applicantsok])  # values are in logit units
_actual = logit(df['% SHSAT Testers'][applicantsok])  # values are in logit units

xs = _predictions
ys = _actual - _predictions  # residual

plt.figure(figsize=(12, 8))
plt.plot(xs, ys, '.')
plt.axhline(0.0, linestyle='--', color='gray')
plt.xlim(-2.5, 2.5)
plt.ylim(-3.5, 3.5)
plt.title("Residual Plot (logit units)")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals");




mae = median_absolute_error(_actual, _predictions)
mse = mean_squared_error(_actual, _predictions)

print("Median Absolute Error:", mae)
print("Mean Squared Error:", mse)




xs = predictions[applicantsok]
ys = df['% SHSAT Testers'][applicantsok]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(xs, ys, s=5)
ax.plot([0, 1], [0, 1], '--', c='gray')
ax.xaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
ax.yaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Regression Results")
ax.set_xlabel("Estimated Percentage of SHSAT Applicants")
ax.set_ylabel("Actual Percentage of SHSAT Applicants");




df_export = predictions.to_frame()
df_export.to_csv("expected_testers.csv")
df_export.head()

