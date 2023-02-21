#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
import seaborn as sns
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')




import plotly
plotly.tools.set_credentials_file(username='....', api_key='......')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode()




import plotly.plotly as py
import plotly.graph_objs as go




df = open('......Credit Fraud Detection/creditcard.csv')
df = pd.read_csv(df)
df.head()




df.shape




df.columns[df.isnull().any()].tolist() 
# there is no missing values in columns




# drop the Time column.since it doesn't make sense in modeling.
df = df.drop(['Time'],axis=1) 




sns.boxplot(df.Amount);
# There are some outliers in the Amount column, however here I will leave them here 




df[df["Amount"] > 10000]
# none of these 7 records is claassified as "fraud".




count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
count_classes = pd.DataFrame(count_classes)
count_classes




labels = ["Legitimate", "Fraud"]
values = count_classes["Class"].tolist()

trace = go.Pie(labels=labels, 
               values=values, 
               textfont=dict(size=18)
              )

py.image.ishow([trace])




from sklearn.cross_validation import train_test_split
np.random.seed(37)
x = df.iloc[:, df.columns != 'Class']
y = df.iloc[:, df.columns == 'Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)




count_fraud = len(y_train.ix[y_train.Class == 1, :])
count_normal = len(y_train.ix[y_train.Class == 0, :])
print count_fraud, count_normal # unbalanced training data




normal_indices = y_train[y_train.Class == 0].index
fraud_indices = y_train[y_train.Class == 1].index

# randomly select "normal" records
np.random.seed(307)
random_normal_indices = np.random.choice(normal_indices, count_fraud, replace = False)
undersample_indices = np.concatenate([fraud_indices,random_normal_indices])




train_undersampled = df.iloc[undersample_indices]




feature_set_1 = pd.concat([train_undersampled.iloc[:, 0:15], train_undersampled["Class"]], axis=1)
feature_set_2 = train_undersampled.iloc[:, 15:31]




sns.pairplot(feature_set_1, hue = "Class");




sns.pairplot(feature_set_2, hue = "Class");




x_train_undersample = x_train.ix[undersample_indices, :]
y_train_undersample = y_train.ix[undersample_indices, :]




from sklearn.model_selection import GridSearchCV




from sklearn.linear_model import LogisticRegressionCV




logit = LogisticRegressionCV(penalty='l1', cv = 10, solver="liblinear")
logit = logit.fit(x_train_undersample, y_train_undersample.values.ravel())      
print "The best parameter C is",logit.C_[0]




ceof_ = logit.coef_.tolist()[0]
feature_names = x_train_undersample.columns.values.tolist()
pd.DataFrame({
    "feature_name": feature_names,
    "values": ceof_
})




from sklearn.ensemble import RandomForestClassifier




rfc = RandomForestClassifier(n_jobs = -1, bootstrap = True, oob_score = True) 

param_grid = { 
    'n_estimators': [50, 100, 200, 500, 700],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

clf_rfc = GridSearchCV(rfc, param_grid=param_grid, cv= 5)
clf_rfc = clf_rfc.fit(x_train_undersample, y_train_undersample.values.ravel())




print 'The best parameters are', clf_rfc.best_params_




from sklearn.metrics import precision_recall_curve, precision_score, roc_auc_score, roc_curve, confusion_matrix, auc, classification_report, recall_score




#normal_indices_complement = map(lambda x: (x in normal_indices) & (x not in random_normal_indices), normal_indices)
np.random.seed(370)
random_normal_indices_1 = np.random.choice(normal_indices, count_fraud, replace = False)
undersample_indices_1 = np.concatenate([fraud_indices,random_normal_indices_1])




x_train_undersample_1 = x_train.ix[undersample_indices_1, :]
y_train_undersample_1 = y_train.ix[undersample_indices_1, :]
#print type(y_train_undersample_1)




# predict with the default threshold 0.5

y_train_pred_logit = logit.predict(x_train_undersample_1.values)
y_train_pred_rfc = clf_rfc.predict(x_train_undersample_1.values)
#print type(y_train_pred_rfc)




import itertools
np.set_printoptions(precision=3)




"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues
                         ):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    plt.grid(False)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




# ROC CURVE
def plot_roc_curve(clf, x_train, y_train):
    
    y_pred_score = clf.predict_proba(x_train)[:,1]
    #print "y_pred_score: "
    #print(y_pred_score)
    #print "y_true: "
    #print(y_train.values.ravel())
    fpr, tpr, thresholds = roc_curve(y_train.values.ravel(), y_pred_score)
    roc_auc = auc(fpr,tpr)
    # Plot ROC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.0])
    plt.ylim([-0.1,1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')




# Precision Recall Curve
def plot_Precision_Recall_Curve(clf, x_train, y_train):
    
    probas_pred = clf.predict_proba(x_train)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_train.values.ravel(), probas_pred)
    #print 'Pricision: ', precision
    pr_auc = auc(recall, precision)
    # Plot Precision-Recall curve
    plt.plot(precision, recall, label='AUC = %0.2f'% pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.03])
    plt.xlim([0.0, 1.03])
    plt.title('Precision-Recall example')
    plt.legend(loc="lower left")




cm_logit = confusion_matrix(y_train_undersample_1,y_train_pred_logit)
cm_rfc = confusion_matrix(y_train_undersample_1,y_train_pred_rfc)
classes_name = ['Legitimate', 'Fraud']




# with the default threshold 0.5, how do those three models work? show with the confusion matrix
figure = plt.figure(figsize=(16, 6))

ax = figure.add_subplot(1,2,1)
plot_confusion_matrix(cm_logit, classes_name)
plt.title("Confusion Matrix of Logit", fontsize = 15)


ax = figure.add_subplot(1,2,2)
plot_confusion_matrix(cm_rfc, classes_name)
plt.title("Confusion Matrix of Random Forest", fontsize = 15);




# ROC with AUC score
figure = plt.figure(figsize=(16,7))

ax = figure.add_subplot(1,2,1)
plot_roc_curve(clf_logit, x_train_undersample_1, y_train_undersample_1)
plt.title("ROC of Logit")

ax = figure.add_subplot(1,2,2)
plot_roc_curve(clf_rfc, x_train_undersample_1, y_train_undersample_1)
plt.title("ROC of Random Forest");




# Precison Recall curve with auc score

figure = plt.figure(figsize=(16,7))

ax = figure.add_subplot(1,2,1)
plot_Precision_Recall_Curve(clf_logit, x_train_undersample_1, y_train_undersample_1)
plt.title('PR Curve of Logit')

ax = figure.add_subplot(1,2,2)
plot_Precision_Recall_Curve(clf_rfc, x_train_undersample_1, y_train_undersample_1)
plt.title('PR Curve of Random Forest');




thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]




def create_cost_df(model, x, y, thresholds):
    results = []
    Amount = x.Amount.tolist()
    for i in range(len(thresholds)):
        y_pred = model.predict_proba(x)[:,1] > thresholds[i]
        y_pred = 1*y_pred
    
        result_i = pd.DataFrame(
            {
                'Amt': Amount,
                't': y.Class.tolist(),
                'p': y_pred
            }
        )
        

        result_i["Ca"] = 2
        result_i["cost"] = result_i.t*(result_i.p*result_i.Ca + (1 - result_i.p)*result_i.Amt) + (1 - result_i.t)*result_i.Ca
        result_i["recall_score"] = recall_score(y, y_pred)
        result_i["precision_score"] = precision_score(y, y_pred)
        
        result_i = result_i.groupby(['recall_score','precision_score'])['cost'].mean().reset_index()
        result_i['threshold'] = thresholds[i]
        
        results.append(result_i)
    
    results = pd.concat(results, axis=0, ignore_index=True)
    return results




cost_df_logit = create_cost_df(logit, x_train_undersample_1, y_train_undersample_1, thresholds)
cost_df_logit




cost_df_rfc = create_cost_df(clf_rfc, x_train_undersample_1, y_train_undersample_1, thresholds)
cost_df_rfc




cost_bar_logit = go.Scatter(x=cost_df_logit.threshold,
                            y=cost_df_logit.cost,
                            name='Toal Cost of Logistic Model',
                            mode = "lines+markers"
                           )

cost_bar_rfc = go.Scatter(x=cost_df_rfc.threshold,
                          y=cost_df_rfc.cost,
                          name='Total Cost of Random Forest Model',
                          mode = "lines+markers"
                         )

layout = go.Layout(title='Total Cost by Models',
                    yaxis=dict(
                        title='Cost in USD',
                        showgrid = True,
                        range = [0,15]
                    ),
                    xaxis=dict(
                        title='Threshold'
                    ),
                    legend=dict(
                       x=0,
                       y=1,
                       bgcolor='rgba(255, 255, 255, 0)',
                       bordercolor='rgba(255, 255, 255, 0)'
                    )
                   )

data = [cost_bar_logit, cost_bar_rfc]

fig = go.Figure(data=data, layout=layout)
py.image.ishow(fig)




recall_logit = go.Scatter(x=cost_df_logit.threshold,
                          y=cost_df_logit.recall_score,
                          name='Recall of Logit',
                          mode = "lines",
                          marker = dict(
                              color = "#E1396C"
                          )
                         )

recall_rfc = go.Scatter(x=cost_df_rfc.threshold,
                        y=cost_df_rfc.recall_score,
                        name='Recall of RF',
                        mode="lines",
                        marker=dict(
                            color = "#FEBFB3"
                        )
                       )
    
precision_logit = go.Scatter(x=cost_df_logit.threshold,
                             y=cost_df_logit.precision_score,
                             name='Precision of Logit',
                             mode = "lines",
                             marker=dict(
                                 color = "rgb(55, 83, 109)"
                             )
                            )


precision_rfc = go.Scatter(x=cost_df_rfc.threshold,
                           y=cost_df_rfc.precision_score,
                           name='Precision of RF',
                           mode = "lines",
                           marker=dict(
                                 color = "rgb(26, 118, 255)"
                             )
                            )

layout = go.Layout(
    title='Model Performance',
    yaxis=dict(
        title='Score'
    ),
    xaxis=dict(
        title='Treshold'
    ),
    barmode = "group"
)
    

data = [recall_logit, precision_logit, recall_rfc, precision_rfc]

fig = go.Figure(data=data, layout=layout)

py.image.ishow(fig)




def Create_Cost_DF(clf, x, y):
    prob_fraud = clf.predict_proba(x)[:,1]
    prob_legitimate = clf.predict_proba(x)[:,0]
    Amount = x.Amount.tolist()
    y_true = y.Class.tolist()
    
    
    cost_df = pd.DataFrame({
        "t": y_true,
        "prob_fraud": prob_fraud,
        "prob_legitimate": prob_legitimate,
        "Amount": Amount
    }
    )
    
    cost_df["Ca"] = 2
    cost_df["risk_fraud"] = cost_df.Ca*cost_df.prob_fraud + cost_df.Ca*cost_df.prob_legitimate
    cost_df["risk_legitimate"] = cost_df.Amount*cost_df.prob_fraud
    
    
    '''
    if risk of classifying as fraud is less than the risk of classifying as legitimate, 
    then classify as fraud. ie. p = 1
    '''
    for i in range(len(y)):
        if cost_df.risk_fraud[i] <= cost_df.risk_legitimate[i]:
            cost_df.ix[i,"p"] = 1
        else:
            cost_df.ix[i,"p"] = 0
    
    cost_df["cost"] = cost_df.t*(cost_df.p*cost_df.Ca + (1 - cost_df.p)*cost_df.Amount) + (1 - cost_df.t)*cost_df.Ca
    
    return cost_df




cost_logit = Create_Cost_DF(logit, x_train_undersample_1, y_train_undersample_1)

recall_logit = recall_score(cost_logit.t, cost_logit.p)
precision_logit = precision_score(cost_logit.t, cost_logit.p)
mean_cost_logit = cost_logit.cost.mean()

print "The recall of logit is %f" % (recall_logit)
print "--------------------------------"
print "The precision of logit is %f" % (precision_logit)
print "--------------------------------"
print "The cost of logit is %f" % (mean_cost_logit)




cost_rfc = Create_Cost_DF(clf_rfc, x_train_undersample_1, y_train_undersample_1)

recall_rfc = recall_score(cost_rfc.t, cost_rfc.p)
precision_rfc = precision_score(cost_rfc.t, cost_rfc.p)
mean_cost_rfc = cost_rfc.cost.mean()

print "The recall of RF is %f" % (recall_rfc)
print "--------------------------------"
print "The precision of RF is %f" % (precision_rfc)
print "--------------------------------"
print "The cost of RF is %f" % (mean_cost_rfc)




'''
The optimal model is rfc with threshold 0.5
'''

y_test_pred = clf_rfc.predict_proba(x_test)[:,1] > 0.5
y_test_pred = 1*y_test_pred

y_test_true = y_test
cm_test = confusion_matrix(y_test_true, y_test_pred)

classes_name = ['Legitimate', 'Fraud']

fig = plt.Figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)
plot_confusion_matrix(cm_test, classes_name,
                          title='Confusion matrix of Test Data'
                         )

print "Recall is %.3f" % (float(cm_test.item(3))/(cm_test.item(3)+cm_test.item(2)))
print "Precision is %.3f" % (float(cm_test.item(3))/(cm_test.item(3)+cm_test.item(1)))

cost_test_df = create_cost_df(clf_rfc, x_test, y_test, thresholds)
print "The cost is %f" % (cost_test_df[cost_test_df.threshold == 0.5]["cost"])




cost_rf_test = Create_Cost_DF(clf_rfc, x_test, y_test)
y_test_pred = cost_rf_test.p
y_test_true = y_test
cm_test_rf = confusion_matrix(y_test_true, y_test_pred)

classes_name = ['Legitimate', 'Fraud']

fig = plt.Figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)
plot_confusion_matrix(cm_test_rf, 
                      classes_name,
                      title='Confusion matrix of Test Data with risk minimum minimization approach'
                     )

print "Recall is %.3f" % (float(cm_test_rf.item(3))/(cm_test_rf.item(3)+cm_test_rf.item(2)))
print "Precision is %.3f" % (float(cm_test_rf.item(3))/(cm_test_rf.item(3)+cm_test_rf.item(1)))
print "The cost is %f" % (cost_rf_test.cost.mean())

