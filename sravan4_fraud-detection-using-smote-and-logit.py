#!/usr/bin/env python
# coding: utf-8



from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV   #Perforing grid search
from imblearn.over_sampling import SMOTE       #over sampling of minority class in imbalanced data
from imblearn.combine import SMOTEENN          #over sampling of minority class in imbalanced data
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

from sklearn.feature_selection import RFE,RFECV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,average_precision_score,precision_recall_curve,precision_score

import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_columns', 100)




# read the input files and look at the top few lines #
data_df= pd.read_csv("../input/creditcard.csv")
data_df.head(2)




data_df['std_amount'] = StandardScaler().fit_transform(data_df['Amount'].values.reshape(-1, 1))
data_df= data_df.drop(['Time','Amount'],axis=1)
data_df.head(2)




response='Class'
pred=data_df.columns[data_df.columns != 'Class']




df_train, df_test = train_test_split(data_df, test_size=0.1, random_state=2,stratify=data_df[response])




# Create the RFE object and compute a cross-validated score.
# Scoring is based on Recall
def rfe_cv(x_train_data,y_train_data):
    rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='recall')
    rfecv.fit(x_train_data, y_train_data)
    
    selected_features=list(x_train_data.columns[rfecv.support_])
    
    print("Optimal number of features: %d" % rfecv.n_features_)
    print('Selected features: %s' % selected_features)

    # Plot number of features VS. cross-validation scores
    plt.figure(figsize=(10,6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (Recall)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    return(selected_features)




predictors=rfe_cv(df_train[pred], df_train[response])




##Function for training models and testing on a test set. if plot is True, it plots a PRC curve for 
##training and test sets and finds the threshold where (precision*recall) is maximum.
def logreg_fit(alg,dtrain,dtest,predictors,response,plot=True):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[response])
        
    #Predict training set:
    dtrain_pred = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    dtest_pred = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
    
    prf_train=precision_recall_fscore_support(dtrain[response], dtrain_pred,pos_label =1, average ='binary')
    prf_test=precision_recall_fscore_support(dtest[response], dtest_pred,pos_label =1, average ='binary')
    area_prc_test=average_precision_score(dtest[response], dtest_predprob,average=None)
    area_prc_train=average_precision_score(dtrain[response], dtrain_predprob,average=None)
    
    
    precision_train, recall_train, thr_train = precision_recall_curve(dtrain[response], dtrain_predprob,pos_label =1)
    precision_test, recall_test, thr_test = precision_recall_curve(dtest[response],dtest_predprob,pos_label =1)
    
    #area, thr = ((precision_test)*(recall_test)), thr_test
    #idx= np.argmax(area)
    
    
    print ("Model report on training data:")
    print ("Train: Precision: %.4g" % (prf_train[0]*100))
    print ("Train: Recall : %.4g" % (prf_train[1]*100))
    print ("Average Precision (Train): %f" % (area_prc_train*100))
    print ("\nModel report on test data:")
    print ("Test: Precision: %.4g" % (prf_test[0]*100))
    print ("Test: Recall : %.4g" % (prf_test[1]*100))
    print ("Average Precision (Trest): %f" % (area_prc_test*100))

    if plot:
        plt.figure(figsize=(10,6))
        plt.plot(recall_test, precision_test, color='coral',label=' Test PR-Curve')
        plt.plot(recall_train, precision_train, color='green',label=' Train PR-Curve')
        plt.plot([0, 1], [0, 1], 'k--')
        #plt.plot([0,recall_test[idx]], [precision_test[idx],precision_test[idx]], 'k--', color='blue')
        #plt.plot([recall_test[idx],recall_test[idx]], [0,precision_test[idx]], 'k--', color='blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()




#Function for K-Fold Stratified Cross_validation of any model
def cv_modelfit(alg,dtrain,predictors,response,cvfolds):

    skf = StratifiedKFold(n_splits=cvfolds,random_state=2)
    cv_results = np.zeros((cvfolds, 3))
    for i, (train_index, test_index) in enumerate(skf.split(dtrain[predictors],dtrain[response])):
        for j in range(0,3):
            cvtrain=dtrain.iloc[train_index]
            cvtest=dtrain.iloc[test_index]

            #Fit the algorithm on the data
            alg.fit(cvtrain[predictors], cvtrain[response])

            #Predict training set:
            dtrain_pred = alg.predict(cvtrain[predictors])
            dtrain_predprob = alg.predict_proba(cvtrain[predictors])[:,1]

            dtest_pred = alg.predict(cvtest[predictors])
            dtest_predprob = alg.predict_proba(cvtest[predictors])[:,1]

            prf_train=precision_recall_fscore_support(cvtrain[response], dtrain_pred,pos_label =1, average ='binary')
            prf_test=precision_recall_fscore_support(cvtest[response], dtest_pred,pos_label =1, average ='binary')
            area_prc_test=average_precision_score(cvtest[response], dtest_predprob,average=None)
            area_prc_train=average_precision_score(cvtrain[response], dtrain_predprob,average=None)


            cvr=[prf_test[0],prf_test[1],area_prc_test]
            cv_results[i,j]=cvr[j]

    print ("Mean CV Test Precision: %.4g" % (cv_results[:,0].mean()*100))
    print ("Mean CV Test Recall: %.4g" % (cv_results[:,1].mean()*100))
    print ("Std.Devation of CV Test Precision: %.4g" % (cv_results[:,0].std()*100))
    print ("Std.Devation of CV Test Recall: %.4g" % (cv_results[:,1].std()*100))




train, test = train_test_split(df_train,test_size=0.1, random_state=2,stratify=df_train[response])




logreg1 = LogisticRegression(penalty = 'l1', C=1,random_state = 2)




logreg_fit(logreg1,train,test,predictors,response,True)




cv_modelfit(logreg1,df_train,predictors,response,10)




##Function for training models and testing on a test set. if plot is True, it plots a PRC curve for 
##training and test sets and finds the threshold where (precision*recall) is maximum.
def logreg_smote(alg,sam_alg,data,predictors,response,plot=True):

    dtrain, dtest = train_test_split(data, test_size=0.1, random_state=2,stratify=data[response])

    
    X_smt, y_smt=sam_alg.fit_sample(train[predictors],train[response])
    X_smt=pd.DataFrame(X_smt)
    columns=train[predictors].columns
    X_smt.columns=columns

    #Fit the algorithm on the data
    alg.fit(X_smt[predictors], y_smt)

    #Predict training set:
    dtrain_pred = alg.predict(X_smt[predictors])
    dtrain_predprob = alg.predict_proba(X_smt[predictors])[:,1]

    #a=float(len(data[data[response]==1]))/float(len(data[response]))
    #b=float(len(y_smt[y_smt==1]))/float(len(y_smt))
    #k=(b/(1-b))*((1-a)/a)

    dtest_pred = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]#/k  

    prf_train=precision_recall_fscore_support(y_smt, dtrain_pred,pos_label =1, average ='binary')
    prf_test=precision_recall_fscore_support(dtest[response], dtest_pred,pos_label =1, average ='binary')
    area_prc_test=average_precision_score(dtest[response], dtest_predprob,average=None)
    area_prc_train=average_precision_score(y_smt, dtrain_predprob,average=None)


    precision_train, recall_train, thr_train = precision_recall_curve(y_smt, dtrain_predprob,pos_label =1)
    precision_test, recall_test, thr_test = precision_recall_curve(dtest[response],dtest_predprob,pos_label =1)

    #area, thr = ((precision_test)*(recall_test)), thr_test
    #idx= np.argmax(area)
    
    print ("Model report on training data:")
    print ("Train: Precision: %.4g" % (prf_train[0]*100))
    print ("Train: Recall : %.4g" % (prf_train[1]*100))
    print ("Average Precision (Train): %f" % (area_prc_train*100))
    print ("\nModel report on test data:")
    print ("Test: Precision: %.4g" % (prf_test[0]*100))
    print ("Test: Recall : %.4g" % (prf_test[1]*100))
    print ("Average Precision (Trest): %f" % (area_prc_test*100))

    
    if plot:

        plt.figure(figsize=(10,6))
        plt.plot(recall_test, precision_test, color='coral',label=' Test PR-Curve')
        plt.plot(recall_train, precision_train, color='green',label=' Train PR-Curve')
        plt.plot([0, 1], [0, 1], 'k--')
        #plt.plot([0,recall_test[idx]], [precision_test[idx],precision_test[idx]], 'k--', color='blue')
        #plt.plot([recall_test[idx],recall_test[idx]], [0,precision_test[idx]], 'k--', color='blue')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()




sampling1=SMOTE(ratio=1.0, random_state=2, k_neighbors=5, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=-1)




logreg_smote(logreg1,sampling1,df_train,predictors,response,plot=True)




def smote_eval(alg,data,predictors,response,r):

    dtrain, dtest = train_test_split(data, test_size=0.1, random_state=2,stratify=data[response])

    smt=SMOTE(ratio=r, random_state=2, k_neighbors=5, 
           m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=2)
    X_smt, y_smt=smt.fit_sample(train[predictors],train[response])
    X_smt=pd.DataFrame(X_smt)
    columns=train[predictors].columns
    X_smt.columns=columns

    #Fit the algorithm on the data
    alg.fit(X_smt[predictors], y_smt)
            
    dtest_pred = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]#/k  

    prf_test=precision_recall_fscore_support(dtest[response], dtest_pred,pos_label =1, average ='binary')
    #area_prc_test=average_precision_score(dtest[response], dtest_predprob,average=None)  
    return(prf_test[0],prf_test[1])




smt_ratio = np.zeros((10, 3))
for i in range(0,10,1):
    for j in range(0,3,1):
        temp=np.append(smote_eval(logreg1,df_train,predictors,response,float(i+1)/float(10)),float(i+1)/float(10))
        smt_ratio[i,j]=temp[j]




smt_ratio=pd.DataFrame(smt_ratio)
smt_ratio.columns=['Precision','Recall','Ratio']
smt_ratio




sampling2=SMOTE(ratio=0.1, random_state=2, k_neighbors=5, 
           m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=-1)




logreg_smote(logreg1,sampling2,df_train,predictors,response,plot=True)




#Function for K-Fold Stratified Cross_validation of any model
def cv_smotemodel(alg,sam_alg,dtrain,predictors,response,cvfolds):

    skf = StratifiedKFold(n_splits=cvfolds,random_state=2)
    cv_results = np.zeros((cvfolds, 3))
    for i, (train_index, test_index) in enumerate(skf.split(dtrain[predictors],dtrain[response])):
        for j in range(0,3):
            cvtrain=dtrain.iloc[train_index]
            cvtest=dtrain.iloc[test_index]
            
            X_smt, y_smt=sam_alg.fit_sample(cvtrain[predictors], cvtrain[response])
            X_smt=pd.DataFrame(X_smt)
            columns=train[predictors].columns
            X_smt.columns=columns
            

            #Fit the algorithm on the data
            alg.fit(X_smt[predictors], y_smt)

            #Predict training set:
            dtrain_pred = alg.predict(X_smt[predictors])
            dtrain_predprob = alg.predict_proba(X_smt[predictors])[:,1]

            dtest_pred = alg.predict(cvtest[predictors])
            dtest_predprob = alg.predict_proba(cvtest[predictors])[:,1]

            prf_train=precision_recall_fscore_support(y_smt, dtrain_pred,pos_label =1, average ='binary')
            prf_test=precision_recall_fscore_support(cvtest[response], dtest_pred,pos_label =1, average ='binary')
            area_prc_test=average_precision_score(cvtest[response], dtest_predprob,average=None)
            area_prc_train=average_precision_score(y_smt, dtrain_predprob,average=None)


            cvr=[prf_test[0],prf_test[1],area_prc_test]
            cv_results[i,j]=cvr[j]

    print ("Mean CV Test Precision: %.4g" % (cv_results[:,0].mean()*100))
    print ("Mean CV Test Recall: %.4g" % (cv_results[:,1].mean()*100))
    print ("Std.Devation of CV Test Precision: %.4g" % (cv_results[:,0].std()*100))
    print ("Std.Devation of CV Test Recall: %.4g" % (cv_results[:,1].std()*100))




cv_smotemodel(logreg1,sampling2,df_train,predictors,response,10)




def grid_tune(param_test,scores):
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print

        clf = GridSearchCV(LogisticRegression(penalty = 'l1', C=1,random_state = 2), param_test, cv=5,
                           scoring=score)
        smt=SMOTE(ratio=0.1, random_state=2, k_neighbors=5, 
           m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=2)
        X_smt, y_smt=smt.fit_sample(train[predictors],train[response])
        X_smt=pd.DataFrame(X_smt)
        columns=train[predictors].columns
        X_smt.columns=columns
        
        clf.fit(X_smt[predictors], y_smt)

        print("Best parameters set found on development set:")
        print
        print(clf.best_params_,clf.best_score_)
        print
        print("Grid scores on development set:")
        print
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))




# Set to n_estimators to the tuned values from cross-validation
scores1 = ['recall','precision']
param_test1 = {
 'penalty':['l1','l2'],
 'C':[0.001, 0.01, 0.1, 1, 10, 100]
}
grid_tune(param_test1,scores1)




# Set to n_estimators to the tuned values from cross-validation
scores1 = ['recall','precision']
param_test2 = {
 'penalty':['l1','l2'],
 'C':[i/100.0 for i in range(1,10)]
}
grid_tune(param_test2,scores1)




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




sampling2=SMOTE(ratio=0.1, random_state=2, k_neighbors=5, 
       m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=2)

X_smt, y_smt=sampling2.fit_sample(df_train[predictors],df_train[response])
X_smt=pd.DataFrame(X_smt)
columns=df_train[predictors].columns
X_smt.columns=columns

#Fit the algorithm on the data
alg = LogisticRegression(penalty = 'l1', C=0.06,random_state = 2)
alg.fit(X_smt[predictors], y_smt)

#Predict test set:
dtest_pred = alg.predict(df_test[predictors])
dtest_predprob = alg.predict_proba(df_test[predictors])[:,1]


prf_test=precision_recall_fscore_support(df_test[response], dtest_pred,pos_label =1, average ='binary')
area_prc_test=average_precision_score(df_test[response], dtest_predprob,average=None)

# Compute confusion matrix
cnf_matrix = confusion_matrix(df_test[response],dtest_pred)
#np.set_printoptions(precision=2)

print ("Test Recall: ",round((prf_test[1]*100),2))
print ("Test Precision: ",round((prf_test[0]*100),2))
print ()

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure(figsize=(7,5))
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

