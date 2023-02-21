#!/usr/bin/env python
# coding: utf-8



## Importing Packages

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import scipy.stats as stats
import os, sys, operator, warnings


# Scikit-learn Auxiliary Modules
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix
from sklearn.metrics import explained_variance_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_curve, precision_score, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, learning_curve, StratifiedKFold, train_test_split, validation_curve 
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline 


# Scikit-learn Classification Models
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# Natural Language Processing
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from textblob import TextBlob, Word, WordList 



# Plotly 
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

py.offline.init_notebook_mode(connected=True)

# Other imports
import itertools
# import pprint
import patsy

# Setting some styles and options
sns.set_style('whitegrid') 
pd.options.display.max_columns = 40 

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
 
get_ipython().run_line_magic('matplotlib', 'inline')

print('Packages Imported Successfully!')




data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', low_memory = False, encoding='ISO-8859-1')

print('Data Loaded Successfuly!')





print('The dataset documents', data.shape[0], 'terror attacks with', data.shape[1], 'different features')




data_columns = [
    
    ## Spatio-Temporal Variables:
                'iyear', 'imonth', 'iday', 'latitude', 'longitude',
    
    ## Binary Variables: 
                'extended', 'vicinity', 'crit1', 'crit2', 'crit3', 'doubtterr',
                'multiple', 'success', 'suicide', 'guncertain1', ## check back guncertain
                'claimed', 'property', 'ishostkid',
    
    ## Continuous Variables:
                'nkill', 'nwound',               
    
    ## Categorical variables (textual): 
                'country_txt', 'region_txt', 'alternative_txt', 'attacktype1_txt', 'targtype1_txt',
                'natlty1_txt', 'weaptype1_txt', 
    
    ## Descriptive Variables: 
                'target1', 'gname', 'summary',    
    
                                            ]

gtd = data.loc[:, data_columns]

# To avoid confusion, we restrict the dataset to only attacks that were of terrorist nature.

gtd = gtd[(gtd.crit1 == 1) & (gtd.crit2 == 1) & (gtd.crit3 == 1) & (gtd.doubtterr == 0)]




gtd.describe()




print ('9/11 attacks:')
gtd[(gtd.iyear == 2001) & (gtd.imonth == 9) & (gtd.iday == 11) & (gtd.country_txt == 'United States')]




gtd.weaptype1_txt.replace(
    'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)',
    'Vehicle', inplace = True)




gtd.iloc[:,[6, 15, 16, 17]] = gtd.iloc[:,[6, 15, 16, 17]].replace(-9,0)




gtd.claimed.replace(2,1, inplace = True) # (3)




gtd.target1 = gtd.target1.str.lower()
gtd.gname = gtd.gname.str.lower()
gtd.summary = gtd.summary.str.lower()    
gtd.target1 = gtd.target1.fillna('unknown').replace('unk','unknown')




gtd.nkill = np.round(gtd.nkill.fillna(gtd.nkill.median())).astype(int) 
gtd.nwound = np.round(gtd.nwound.fillna(gtd.nwound.median())).astype(int) 




gtd['casualties'] = gtd.nkill + gtd.nwound
gtd['nclass'] = gtd.casualties.apply(lambda x: 0 if x == 0 else 1) 




def categorize_perpetrators(column):
    '''
    This function reorganizes perpetrator groups based on their value_counts, perpetrator groups with
    less than 10 occurences are re-assigned to a new category called 'small_time_perpetrator'
    Parameter is of the type <pandas.core.series.Series>
    '''
    perpetrators_count = column.value_counts()
    small_time_perpetrator = perpetrators_count[perpetrators_count < 10].index.tolist()
    column = column.apply(lambda x: 'small time perpetrator' if x in small_time_perpetrator else x).astype(str)
    return column





gtd.gname = categorize_perpetrators(gtd.gname)
print('Perpetrators categorized!')




def categorize_target1(column):
    '''
    This function performs three operations:
    - It uses TextBlop in order to lemmatize (e.g. transform a word into its cannonical form) the textual data,
    for example, converting 'civilians' to 'civilian'. This enables us to increase the value count for recurrent
    words.
    - The second part of the function defines a list of top_targets, which include targets mentioned more than
    50 times. It then loops through every target string and re-assigns sentences that contain top_targets words.
    - Finally, it assigns every target not in top_targets to a new 'isolated target' category.
    Parameter is of the type <pandas.core.series.Series>
    '''
    
    temp_target = []
    for target in column:
        blob = TextBlob(target)
#         blob.ngrams = 2
        blop = blob.words
        lemma = [words.lemmatize() for words in blop]
        temp_target.append(" ".join(lemma))
    column = pd.Series(temp_target, index = column.index)
    target_count = column.value_counts()
    top_targets = target_count[target_count > 50].index.tolist()
    for item in top_targets: 
        column = column.apply(lambda x: item if item in x else x)
    column = column.apply(lambda x: 'isolated target' if x not in top_targets else x)
    return column




gtd.target1 = categorize_target1(gtd.target1)
print('Targets categorized!')




print ('missing data : \n')
print (gtd.drop(['latitude','longitude','summary'], axis = 1).isnull().sum().sort_values(ascending = False).head(4))




df = gtd.drop(['longitude','latitude', 'summary'], axis =1)




df.shape




df.guncertain1.fillna(0, inplace = True)
df.ishostkid.fillna(0, inplace = True)




y_temp = df.claimed
y_temp.shape




categorical = ['country_txt', 'alternative_txt', 'attacktype1_txt',
               'targtype1_txt', 'weaptype1_txt', 'gname', 'target1']

numerical = ['extended', 'vicinity', 'multiple', 'success',
             'suicide', 'guncertain1', 'casualties', 'property', 'ishostkid',]




formula =  ' + '.join(numerical)+ ' + ' + ' + '.join(['C('+i+')' for i in categorical]) + ' -1' 
formula




X_temp = patsy.dmatrix(formula, data = df, return_type= 'dataframe')
print(X_temp.shape, y_temp.shape)




X_train = X_temp[~y_temp.isnull()]
X_test = X_temp[y_temp.isnull()]




y_train = y_temp[~y_temp.isnull()]
y_test = y_temp[y_temp.isnull()]




X_train.shape, y_train.shape, X_test.shape, y_test.shape




lr = LogisticRegression(random_state = 42).fit(X_train, y_train) 




predictions = pd.Series(lr.predict(X_test), index = X_test.index)




df.claimed.fillna(predictions, inplace = True)




# imputed_values = [ pred + sampling_from_norma_distribution for pred in predicted]




df = pd.read_csv('./Assets/modeling.csv') 
print ('Data Loaded Successfuly!')




trace = dict(
    type = 'choropleth',
    locationmode = 'country names',
    locations = cpc['country_txt'],

    z = cpc['casualties'],
    name = 'Casualties',
    text = cpc['country_txt'].astype(str) + '<br>' + cpc['casualties'].astype(str),
    hoverinfo = 'text+name',
    autocolorscale = False,
    colorscale = 'Viridis',
#     reversescale = True,
    marker = dict( line = dict ( color = 'rgb(255,255,255)', width = 0.5))
        
    )
        

layout = dict(
    title = 'Cummulative Casualties World Map from 1970 until 2015 ',
    geo = dict( showframe = False, showcoastlines = True,
               projection = dict(type = 'Mercator'), showlakes = True,
               lakecolor = 'rgb(255, 255, 255)'       
              )
    )
    

py.iplot(dict( data=[trace], layout=layout ))




cpy = df.groupby('iyear', as_index=False)['casualties'].sum()

trace = go.Scatter(x = cpy.iyear, y = cpy.casualties,
                   name = 'Casualties', line = dict(color = 'salmon', width = 4, dash ='dot'),
                   hoverinfo = 'x+y+name')

layout = go.Layout(title = 'Casualties per Year')

py.iplot(dict(data = [trace], layout = layout))     





cpr = df.groupby('region_txt', as_index= False)['casualties'].sum()
apr = df.groupby('region_txt')['region_txt'].count()

trace_1 = go.Bar(x = cpr.region_txt, y = cpr.casualties,
                 marker = dict(color = 'rgb(100, 229, 184)'),
                 name = 'Casualties')

trace_2 = go.Bar(x = apr.index, y = apr,
                 marker = dict(color = 'rgb(255, 188, 214)'),
                 name = 'Terror Attacks')

layout = go.Layout(title = "Total Casualties and Terror Attacks by Region", barmode='group' )


py.iplot(dict(data = [trace_1,trace_2], layout = layout))




### Top 10 countries by attack/fatalities
apc = df.groupby('country_txt')['country_txt'].count().sort_values(ascending= False)
cpc = df.groupby('country_txt', as_index= False)['casualties'].sum().sort_values(by = 'casualties', ascending= False)
cc = pd.merge(pd.DataFrame(apc), cpc, on = 'country_txt')


trace = go.Bar(x = apc.index[:20],y = apc,
                 marker = dict(color = 'rgb(255, 188, 214)'),
                 name = 'Terror Attacks')

layout = go.Layout(title = 'top 20 most targeted countries', barmode='relative' )

py.iplot(dict(data = [trace], layout = layout)) 




#### Notes to self

## Are nationalities more likely to get killed?
## Are certain countries better at capturing perpetrators?
## Are there countries particularily focused on kidnappings/hostages?
## How well do certain countries defend against terrorist attacks?
# 3d filled lines for number of attacks per year per top country
# Fatalities by target




y = df.casualties.apply(lambda x: 0 if x == 0 else 1).values





numerical = ['extended', 'vicinity', 'multiple', 'success', 'claimed',
             'suicide', 'guncertain1', 'property', 'ishostkid','natlty1_txt']

categorical = ['country_txt', 'alternative_txt', 'attacktype1_txt',
              'targtype1_txt', 'weaptype1_txt', 'gname', 'target1']




formula =  ' + '.join(numerical)+ ' + ' + ' + '.join(['C('+i+')' for i in categorical]) + ' -1' 
formula




X = patsy.dmatrix(formula, data = df, return_type= 'dataframe')




print X.shape, y.shape




X.head(2) 




pca_model = PCA(n_components=len(X.columns)) 
pca = pca_model.fit(X)




var_ratio = pca.explained_variance_ratio_
var_ratio = np.cumsum(var_ratio)
plot_cumsum_variance(var_ratio)




X_columns = list(X.columns) #Here we transfrom our variables into a list

#We then apply a chi2 statistical measure
skb_chi2 = SelectKBest(chi2, k=20)
skb_chi2.fit(X, y)

# examine results
top_15_chi2 = pd.DataFrame([X_columns, list(skb_chi2.scores_)], 
                     index=['feature','chi2 score']).T.sort_values('chi2 score', ascending=False)[:15]
top_15_chi2




plt.figure(figsize=(13,6))

sns.barplot(x = top_15_chi2['chi2 score'], y = top_15_chi2.feature, palette= 'viridis')
plt.show()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 102)

print X_train.shape, y_train.shape, X_test.shape, y_test.shape




vanilla_models = { 
    
    # Linear Models
    'Logistic Regression' : LogisticRegression(n_jobs = -1, random_state = 56, penalty = 'l1'),
    'Perceptron' : Perceptron(n_iter = 20, n_jobs = -1, random_state= 56),
    'SGD Classifier' : SGDClassifier(penalty = 'l1', n_jobs = -1, random_state= 56),
    
    # Support Vector Machine
    'Linear SVC' : LinearSVC(penalty = 'l1', random_state = 56, dual = False),
    
    # Naive Bayes:
    'Gaussian Naive-Bayes' : GaussianNB(),
    
    # Decision Tree & Ensemble
    'Decision Tree Classifier' : DecisionTreeClassifier(random_state= 56),
    'Random Forest Classifier': RandomForestClassifier(n_jobs = -1, random_state= 56),
    'Gradient Boosting Classifier' : GradientBoostingClassifier(random_state= 56),
    'AdA Boost Classifier': AdaBoostClassifier(random_state = 56),
    'Bagging Classifier' : BaggingClassifier(random_state= 56, n_jobs = -1),
    
    # K-Nearest Neighbor:
    
    # Multi-Layer Perceptron (Neural Network):
    'MLP Classifier' : MLPClassifier(activation = 'logistic', random_state = 56, max_iter=400),  
    
}




score_table = pd.DataFrame(columns = ['model', 'cv_10'])


for model, n in zip(vanilla_models, np.arange(len(vanilla_models))):
                    
    clf = Pipeline([
          ('classification', vanilla_models[model]),
        ])
    
    clf.fit(X_train, y_train)
    
    cv_10 = cross_val_score(clf, X_test, y_test, cv = 10, scoring = 'recall').mean()
    
    score_table.loc[n,'model'] = model
    score_table.loc[n,'cv_10'] = cv_10




score_table.sort_values(by = 'cv_10', ascending = False)





plt.figure(figsize = (11,4))
plt.xticks(rotation = 45, ha = 'center')
sns.barplot(score_table.model, score_table.cv_10, palette = 'viridis');




lr = LogisticRegression(random_state = 56, n_jobs = -1, penalty = 'l1')

lr_params = {
    'C': np.linspace(0.001, 1, 20),
}




lr_grid = GridSearchCV(lr, lr_params, scoring = 'recall', cv = 5, n_jobs = -1, error_score = 0)




lr_grid.fit(X_train, y_train)




lr_best_estimator = lr_grid.best_estimator_

print 'best estimator: \n', lr_grid.best_estimator_

print '\naccuracy_score: \n', lr_grid.score(X_test, y_test)

print '\nbest_params: \n', lr_grid.best_params_




lr_results = pd.DataFrame(lr_grid.cv_results_).sort_values(by = 'param_C')




lr_results.head(3)




lr_results.plot(x ='param_C', y = 'mean_test_score');




lr_score = cross_val_score(lr_grid.best_estimator_, X_test, y_test, cv = 10, scoring = 'recall').mean()
lr_score









# lr_coef = pd.DataFrame(lr_best_estimator.coef_, columns = X.columns).T.sort_values(by = 0, ascending = False).rename(columns = {0: 'coef'})




# sns.barplot(x=lr_coef[:10].coef, y=lr_coef[:10].index)




sgd = SGDClassifier(random_state = 56, n_jobs = -1, n_iter = 200, penalty = 'elasticnet', l1_ratio = 0.01)

sgd_params = {

    'alpha' : np.logspace(-5,0, 10)
    
}




sgd_grid = GridSearchCV(sgd, sgd_params, cv = 5, scoring = 'recall', n_jobs = -1, error_score = 0, random_state=212)




sgd_grid.fit(X_train, y_train)




sgd_best_estimator = sgd_grid.best_estimator_ 

print sgd_grid.best_estimator_
print
print sgd_grid.score(X_test, y_test)
print 
print sgd_grid.best_params_




sgd_results = pd.DataFrame(sgd_grid.cv_results_).sort_values(by = 'param_alpha')




sgd_results.head(3)




sgd_results.plot(x= 'param_alpha', y = 'mean_test_score', logx=True);




sgd_score = cross_val_score(sgd_grid.best_estimator_, X_test, y_test, cv = 10, scoring = 'recall', n_jobs = -1).mean()
sgd_score









svm = LinearSVC(random_state = 56, penalty = 'l1', dual = False)
svm_params = {
    
    'C': np.linspace(0.001, 10, 15),
    
}




svm_grid = GridSearchCV(svm, svm_params, cv = 5, scoring = 'recall', n_jobs = -1, error_score = 0)




warnings.filterwarnings('ignore')

svm_grid.fit(X_train, y_train)




warnings.filterwarnings('default')




svm_best_estimator = svm_grid.best_estimator_
get_ipython().run_line_magic('store', 'svm_best_estimator')
print svm_grid.best_estimator_
print
print svm_grid.score(X_test, y_test)

print svm_grid.best_params_




svm_results = pd.DataFrame(svm_grid.cv_results_).sort_values(by = 'param_C')




svm_results.head(3)




svm_results.plot(x = 'param_C', y = 'mean_test_score');




svm_score = cross_val_score(svm_grid.best_estimator_, X_test, y_test, scoring = 'recall', cv = 10, n_jobs = -1).mean()
svm_score




get_ipython().run_line_magic('store', 'svm_score')
get_ipython().run_line_magic('store', 'svm_results')




rf = RandomForestClassifier(random_state = 56, n_jobs = -1, n_estimators= 300)

rf_params = {
    
    'criterion': ['gini','entropy'],
    'max_features' : ['auto', 'sqrt'],
    
    
}




rf_grid = GridSearchCV(rf, rf_params, scoring = 'recall', cv = 5, n_jobs = -1, error_score= 0)




rf_grid.fit(X_train, y_train)




rf_best_estimator =rf_grid.best_estimator_
print rf_grid.best_estimator_
print
print rf_grid.score(X_test, y_test)
print
print rf_grid.best_params_




rf_results = pd.DataFrame(rf_grid.cv_results_).sort_values(by = 'rank_test_score')




rf_results.head(3)




get_ipython().run_line_magic('store', 'rf_results')
get_ipython().run_line_magic('store', 'rf_best_estimator')









rf_score = cross_val_score(rf_grid.best_estimator_, X_test, y_test, cv = 10, scoring = 'recall', n_jobs = -1).mean()
rf_score




get_ipython().run_line_magic('store', '-r rf_best_estimator')




mlp = MLPClassifier(

    hidden_layer_sizes= (40,), 
    activation = 'logistic',
    learning_rate = 'adaptive',
    learning_rate_init = 0.2,
    random_state = 56,
    max_iter = 500,    
    
)

mlp_params = {
#     'hidden_layer_sizes' : [10, 20, 30, 40, 50],
    'alpha' : np.logspace(-5,1,10),
    'solver' : ['adam', 'sgd'],
#     'solver' : ['adam','sgd'],
#     'learning_rate_init' : [0.2, 0.0001],
       
    }







mlp_grid = GridSearchCV(mlp, mlp_params, scoring = 'recall', cv = 5, n_jobs = -1, error_score= 0)




# mlp_grid.fit(X_train, y_train)




print mlp_grid.best_estimator_
print mlp_grid.score(X_test, y_test)
print mlp_grid.best_params_




mlp_best_estimator = mlp_grid.best_estimator_
mlp_best_estimator




get_ipython().run_line_magic('store', 'mlp_best_estimator')
get_ipython().run_line_magic('store', 'mlp_results')









mlp_results.sort_values(by = 'param_alpha', inplace = True, ascending = False)




mlp_results[mlp_results.param_solver == 'adam'].plot(x = 'param_alpha', y = 'mean_test_score', logx = True);




mlp_results[mlp_results.param_solver == 'sgd'].plot(x = 'param_alpha', y = 'mean_test_score', logx = True);




mlp_score = cross_val_score(mlp_best_estimator, X_test, y_test, cv = 10, scoring = 'recall', n_jobs = -1).mean()
mlp_score














score_table = pd.DataFrame(columns = ['model', 'cv_10_score'])
models = ['Logistic Regression', 'SGD Classifier', 'SVC Classifier', 'Random Forest Classifier', 'Multi-Layer Perceptron']
score_list = [lr_score, sgd_score, svm_score, rf_score,  mlp_score]

for model, n, score in zip(models, np.arange(len(models)), score_list):
    score_table.loc[n,'model'] = model
    score_table.loc[n,'cv_10_score'] = score           




score_table





plt.figure(figsize = (11,4))
plt.xticks(rotation = 45, ha = 'center')
sns.barplot(score_table.model, score_table.cv_10_score, palette = 'viridis');




#Rank by mean coefficient value across models
# coef_table = pd.DataFrame(zip(lasso.coef_[0],lr.coef_[0],X_train.columns), columns = ['Lasso_coef','Ridge_coef','Features'])
# coef_table.head(10)




plot_confusion_matrix(confusion_matrix(y_test, lr_grid.best_estimator_.predict(X_test)), title = 'Logistic Regression', classes = np.array([0,1]))
plot_confusion_matrix(confusion_matrix(y_test, sgd_grid.best_estimator_.predict(X_test)), title = 'SGD Classifier', classes = np.array([0,1]))
plot_confusion_matrix(confusion_matrix(y_test, svm_grid.best_estimator_.predict(X_test)), title = 'SVC Classifier', classes = np.array([0,1]))
plot_confusion_matrix(confusion_matrix(y_test, rf_grid.best_estimator_.predict(X_test)), title = 'Random Forest Classifier', classes = np.array([0,1]))
plot_confusion_matrix(confusion_matrix(y_test, mlp_grid.best_estimator_.predict(X_test)), title = 'Multi-Layer Perceptron', classes = np.array([0,1]))




plt.figure(1)
plot_confusion_matrix(confusion_matrix(y_test, lr_grid.best_estimator_.predict(X_test)), title = 'Logistic Regression', classes = np.array([0,1]))
plt.figure(2)
plot_confusion_matrix(confusion_matrix(y_test, sgd_grid.best_estimator_.predict(X_test)), title = 'SGD Classifier', classes = np.array([0,1]))














plot_roc(lr_grid.best_estimator_, 'Logistic Regression')
# plot_roc(sgd_grid.best_estimator_, 'SGD Classifier')
# plot_roc(svm_grid.best_estimator_, 'SVC Classifier')
plot_roc(rf_grid.best_estimator_, 'Random Forest Classifier')
plot_roc(mlp_best_estimator, 'Multi-Layer Perceptron')




def plot_confusion_matrix(cm, classes, title, cmap='viridis'):
    '''
    This function simply gives a nice loooking layout to the confusion matrix.
    '''
    plt.figure(figsize = (3,3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()




def plot_cumsum_variance(var_ratio):
    '''
    This function plots cummulative explained variance, ranking features by PCA importance.
    '''
    fig = plt.figure(figsize=(15,5))#init figure 
    ax = fig.gca()
    
    x_vals = range(1,len(var_ratio)+1)#set x&y values
    y_vals = var_ratio
    
    ax.set_title('Explained Variance over Principal Components')#set title and labels 
    ax.set_ylabel('Cumulative Sum of Variance Explained')
    ax.set_xlabel('Number of Principal Components')
    
    ax.plot(x_vals, y_vals)




def plot_roc(model, varname):
    y_pp = model.predict_proba(X_test)[:, 1]
    fpr_, tpr_, _ = roc_curve(y_test, y_pp)
    auc_ = auc(fpr_, tpr_)
    acc_ = np.abs(0.5 - np.mean(y)) + 0.5
    
    fig, axr = plt.subplots(figsize=(5,4))

    axr.plot(fpr_, tpr_, label='ROC (area = %0.2f)' % auc_,
             color='darkred', linewidth=4,
             alpha=0.7)
    axr.plot([0, 1], [0, 1], color='grey', ls='dashed',
             alpha=0.9, linewidth=4, label='baseline accuracy = %0.2f' % acc_)

    axr.set_xlim([-0.05, 1.05])
    axr.set_ylim([0.0, 1.05])
    axr.set_xlabel('false positive rate', fontsize=16)
    axr.set_ylabel('true positive rate', fontsize=16)
    axr.set_title(varname+' ROC', fontsize=20)

    axr.legend(loc="lower right", fontsize=12)

    plt.show()









# The end!




# sns.distplot(data.year_salary,bins=60, color ="dimgrey",kde_kws={"color": "darkred", "lw": 1, "label": "KDE"})











