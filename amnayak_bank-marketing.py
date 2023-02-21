#!/usr/bin/env python
# coding: utf-8



# Importing the libraries
import pandas as pd        # for data manipulation
import seaborn as sns      # for statistical data visualisation
import numpy as np         # for linear algebra
import matplotlib.pyplot as plt      # for data visualization
from scipy import stats        # for calculating statistics

# Importing various machine learning algorithm from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,roc_curve,auc,accuracy_score
from  sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier




dataframe= pd.read_csv("bank-full.csv")  # Reading the data
dataframe.head()   # showing first 5 datas




dataframe.shape




dataframe.info()




dataframe.isnull()




dataframe.apply(lambda x: len(x.unique()))




dataframe.describe()




sns.pairplot(dataframe)




dataframe.Target.value_counts()




f = plt.subplots(1, figsize=(12,4))

colors = ["#FA5858", "#64FE2E"]
labels ="Did not Open Term Deposit", "Opened Term Deposit"

plt.suptitle('Information on Term Deposits', fontsize=20)

plt.pie(dataframe.Target.value_counts(),explode=[0,0.25], shadow=True,colors=colors,labels=labels, startangle=25,autopct='%1.1f%%')

plt.show()




plt.figure(figsize=(10,10))
plt.subplot(5,1,1)
sns.boxplot(dataframe.age)
plt.subplot(5,1,2)
sns.boxplot(dataframe.balance)
plt.subplot(5,1,3)
sns.boxplot(dataframe.campaign)
plt.subplot(5,1,4)
sns.boxplot(dataframe.duration)




dataframe.skew()




plt.figure(figsize=(10,10))
plt.subplot(3,1,1)
sns.distplot(dataframe.age)
plt.subplot(3,1,2)
sns.countplot(dataframe.job)
plt.subplot(3,1,3)
sns.countplot(dataframe.marital)
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20,8))
sns.countplot(x = 'default', data = dataframe, ax = ax1)
sns.countplot(x = 'housing', data = dataframe, ax = ax2)
sns.countplot(x = 'loan', data = dataframe, ax = ax3)




fig,(a1,a2)=plt.subplots(nrows = 1, ncols = 2, figsize = (13, 5))
sns.boxplot(x = 'duration', data = dataframe, orient = 'v', ax = a1)
sns.distplot(dataframe.duration, ax = a2)




dataframe[dataframe.duration==0]




dataframe["Target"].hist(bins=2)




dataframe["Target"].value_counts()




plt.figure(figsize=(10,10))
sns.boxplot(x='default',y='balance',data=dataframe,hue='Target',palette='muted')




fig,(a1,a2)=plt.subplots(nrows = 2, ncols = 1, figsize = (13, 15))
sns.boxplot(x='job',y='age',data=dataframe,ax=a1)
sns.boxplot(x='job',y='balance',hue='Target',data=dataframe,ax=a2)




plt.figure(figsize=(10,10))
sns.countplot(x="education", data=dataframe,hue="Target")




sns.catplot(x='duration',y='balance',data=dataframe,hue='marital')




sns.countplot(x='marital',hue='Target',data=dataframe)




sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style('whitegrid')
avg_duration = dataframe['duration'].mean()

lst = [dataframe]
dataframe["duration_status"] = np.nan

for col in lst:
    col.loc[col["duration"] < avg_duration, "duration_status"] = "below_average"
    col.loc[col["duration"] > avg_duration, "duration_status"] = "above_average"
    
pct_term = pd.crosstab(dataframe['duration_status'], dataframe['Target']).apply(lambda r: round(r/r.sum(), 2) * 100, axis=1)


ax = pct_term.plot(kind='bar', stacked=False, cmap='RdBu')
plt.xlabel("Duration Status", fontsize=18);
plt.ylabel("Percentage (%)", fontsize=18)

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))
    

plt.show()




labelencoder_X=LabelEncoder()
dataframe.Target=labelencoder_X.fit_transform(dataframe.Target)
corelation=dataframe.corr()




plt.figure(figsize=(10,10))
a=sns.heatmap(corelation,annot=True)




labelencoder_X = LabelEncoder()
dataframe['job']      = labelencoder_X.fit_transform(dataframe['job']) 
dataframe['marital']  = labelencoder_X.fit_transform(dataframe['marital']) 
dataframe['education']= labelencoder_X.fit_transform(dataframe['education']) 
dataframe['default']  = labelencoder_X.fit_transform(dataframe['default']) 
dataframe['housing']  = labelencoder_X.fit_transform(dataframe['housing']) 
dataframe['loan']     = labelencoder_X.fit_transform(dataframe['loan'])
dataframe['contact']     = labelencoder_X.fit_transform(dataframe['contact']) 
dataframe['day'] = labelencoder_X.fit_transform(dataframe['day']) 
dataframe['month'] = labelencoder_X.fit_transform(dataframe['month']) 




print('1º Quartile: ', dataframe['duration'].quantile(q = 0.25))
print('2º Quartile: ', dataframe['duration'].quantile(q = 0.50))
print('3º Quartile: ', dataframe['duration'].quantile(q = 0.75))
print('4º Quartile: ', dataframe['duration'].quantile(q = 1.00))

print('1º Quartile: ', dataframe['age'].quantile(q = 0.25))
print('2º Quartile: ', dataframe['age'].quantile(q = 0.50))
print('3º Quartile: ', dataframe['age'].quantile(q = 0.75))
print('4º Quartile: ', dataframe['age'].quantile(q = 1.00))




dataframe.loc[dataframe['age'] <= 33, 'age'] = 1
dataframe.loc[(dataframe['age'] > 33) & (dataframe['age'] <= 39), 'age'] = 2
dataframe.loc[(dataframe['age'] > 39) & (dataframe['age'] <= 48), 'age'] = 3
dataframe.loc[(dataframe['age'] > 48) & (dataframe['age'] <= 98), 'age'] = 4

dataframe.loc[dataframe['duration'] <= 103, 'duration'] = 1
dataframe.loc[(dataframe['duration'] > 103) & (dataframe['duration'] <= 180)  , 'duration']    = 2
dataframe.loc[(dataframe['duration'] > 180) & (dataframe['duration'] <= 319)  , 'duration']   = 3
dataframe.loc[(dataframe['duration'] > 319) & (dataframe['duration'] <= 644.5), 'duration'] = 4
dataframe.loc[dataframe['duration']  > 644.5, 'duration'] = 5




dataframe['poutcome'].replace(['unknown', 'failure','other','success'], [1,2,3,4], inplace  = True)
#dataframe['Target'].replace(['no', 'yes'], [0,1], inplace  = True)




dataframe.head()




dataframe.columns




features=['age', 'job', 'marital', 'education', 'default','balance','duration',
       'housing', 'loan', 'contact','month', 'day','campaign','pdays','previous','poutcome']
X=dataframe[features]
Y=dataframe['Target']         




train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=1)
train_X.count() 




train_X.head()




test_X.count()




test_X.head()





scaling = StandardScaler()
train_X = scaling.fit_transform(train_X)
test_X = scaling.fit_transform(test_X)
print(train_X)




LR_Model=LogisticRegression(random_state=1)
Logestic_Model=LR_Model.fit(train_X,train_y)
Logestic_Model




predict=LR_Model.predict(test_X)
print(predict[0:1000])
metrics=confusion_matrix(test_y,predict)
metrics




sns.heatmap(metrics,annot=True,fmt='g',cmap='Blues')




print(classification_report(test_y,predict))




probability=Logestic_Model.predict_proba(test_X)
pred=probability[:,1]
fpr,tpr,thresh=roc_curve(test_y,pred)
roc_auc=auc(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,'b',label='AUC =%0.2f'%roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




LR_accuracy=accuracy_score(test_y,predict)
LR_accuracy




LR_AUC=roc_auc
LR_AUC




LR_Gini = 2*roc_auc - 1
LR_Gini




n=[1,3,5,7,11,13,15,17,19,21,23,25,27,29,31,33,35]
accuracy_scores=[]
for i in n:
    KNN_Model=KNeighborsClassifier(n_neighbors=i)
    KNN_Model.fit(train_X,train_y)
    predict=KNN_Model.predict(test_X)
    accuracy_scores.append(accuracy_score(test_y,predict))
accuracy_scores
    




p=[1,2]
accuracy_scores=[]
for i in p:
    KNN_Model=KNeighborsClassifier(n_neighbors=11,p=i)
    KNN_Model.fit(train_X,train_y)
    predict=KNN_Model.predict(test_X)
    accuracy_scores.append(accuracy_score(test_y,predict))
accuracy_scores

    




KNN_Model=KNeighborsClassifier(n_neighbors=13,p=1)  
KNN_Model.fit(train_X,train_y)
predict=KNN_Model.predict(test_X)
print(predict[0:200,])
Knn_matrics=confusion_matrix(test_y,predict)
Knn_matrics




print(classification_report(test_y,predict))




sns.heatmap(Knn_matrics,annot=True,cmap='Blues',fmt='g')




probs = KNN_Model.predict_proba(test_X)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




KNN_accuracy=accuracy_score(test_y,predict)
KNN_accuracy




KNN_Gini=2*roc_auc-1
KNN_Gini




KNN_AUC=roc_auc
KNN_AUC





NB_Model=GaussianNB()
naiveB_Model=NB_Model.fit(train_X,train_y)
naiveB_Model




predict=NB_Model.predict(test_X)
predict[0:200,]




ac_score=accuracy_score(test_y,predict)
ac_score




print(classification_report(test_y,predict))




NB_matrics=confusion_matrix(test_y,predict)
NB_matrics




sns.heatmap(NB_matrics,annot=True,cmap='Blues',fmt='g')




probs=NB_Model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




NB_accuracy=accuracy_score(test_y,predict)
NB_accuracy




NB_Gini=2*roc_auc-1
NB_Gini




NB_AUC=roc_auc
NB_AUC





svc=SVC(kernel='sigmoid',random_state=1,probability=True)
svc_Model=svc.fit(train_X,train_y)
svc_Model




predict=svc_Model.predict(test_X)
predict[0:200,]




ac_score=accuracy_score(test_y,predict)
ac_score




print(classification_report(test_y,predict))




svc_matrics=confusion_matrix(test_y,predict)
svc_matrics




sns.heatmap(svc_matrics,annot=True,cmap='Blues',fmt='g')




probs=svc_Model.predict_proba(test_X)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




svc_accuracy=accuracy_score(test_y,predict)
svc_accuracy




svc_Gini=2*roc_auc-1
svc_Gini




svc_AUC=roc_auc
svc_AUC




data=[[LR_accuracy,LR_Gini,LR_AUC],[KNN_accuracy,KNN_Gini,KNN_AUC],[NB_accuracy,NB_Gini,NB_AUC],[svc_accuracy,svc_Gini,svc_AUC]]




comparison=pd.DataFrame(data,index=['Logestic','KNN','Naive Bayes','SVC'],columns=['Accuracy','Gini','AUC'])
comparison




dtree_accuracy_score=[]
for m in range(1,18):
    dtree=DecisionTreeClassifier(criterion='gini',max_depth=m,random_state=1)
    dtree_model=dtree.fit(train_X,train_y)
    predict=dtree.predict(test_X)
    dtree_accuracy_score.append(accuracy_score(test_y,predict))
dtree_accuracy_score




dtree=DecisionTreeClassifier(criterion='gini',max_depth=8,random_state=1)
dtree_model=dtree.fit(train_X,train_y)
predict=dtree.predict(test_X)




predict=dtree_model.predict(test_X)
predict[0:200,]




ac_score=accuracy_score(test_y,predict)
ac_score




print(classification_report(test_y,predict))




dtree_matrics=confusion_matrix(test_y,predict)
dtree_matrics




sns.heatmap(dtree_matrics,annot=True,cmap='Blues',fmt='g')




probs=dtree_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




dtree_accuracy=accuracy_score(test_y,predict)
dtree_accuracy




dtree_Gini=2*roc_auc-1
dtree_Gini




dtree_AUC=roc_auc
dtree_AUC




#bagging=BaggingClassifier(n_estimators=50,random_state=1)
bagging_accuracy_score=[]
for m in range(50,90):
    bagging=BaggingClassifier(n_estimators=m,random_state=1)    
    bagging_model=bagging.fit(train_X,train_y)
    predict=dtree.predict(test_X)
    bagging_accuracy_score.append(accuracy_score(test_y,predict))
bagging_accuracy_score




bagging=BaggingClassifier(n_estimators=50,random_state=1)    
bagging_model=bagging.fit(train_X,train_y)
predict=bagging.predict(test_X)




predict=bagging_model.predict(test_X)
predict[0:200,]




ac_score=accuracy_score(test_y,predict)
ac_score




print(classification_report(test_y,predict))




bagging_matrics=confusion_matrix(test_y,predict)
bagging_matrics




sns.heatmap(bagging_matrics,annot=True,cmap='Blues',fmt='g')




probs=bagging_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




bagging_accuracy=accuracy_score(test_y,predict)
bagging_accuracy




bagging_Gini=2*roc_auc-1
bagging_Gini




bagging_AUC=roc_auc
bagging_AUC




aboost_accuracy_score=[]
for m in range(50,90):
    aboost=AdaBoostClassifier(n_estimators=m,random_state=1)    
    aboost_model=aboost.fit(train_X,train_y)
    predict=aboost.predict(test_X)
    aboost_accuracy_score.append(accuracy_score(test_y,predict))
aboost_accuracy_score




aboost=AdaBoostClassifier(n_estimators=66,random_state=1)    
aboost_model=aboost.fit(train_X,train_y)
predict=aboost.predict(test_X)




predict=aboost_model.predict(test_X)
predict[0:200,]




ac_score=accuracy_score(test_y,predict)
ac_score




print(classification_report(test_y,predict))




aboost_matrics=confusion_matrix(test_y,predict)
aboost_matrics




sns.heatmap(aboost_matrics,annot=True,cmap='Blues',fmt='g')




probs=aboost_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




aboost_accuracy=accuracy_score(test_y,predict)
aboost_accuracy




aboost_Gini=2*roc_auc-1
aboost_Gini




aboost_AUC=roc_auc
aboost_AUC




gboost_accuracy_score=[]
for m in range(50,90):
    gboost=GradientBoostingClassifier(n_estimators=m,random_state=1)    
    gboost_model=gboost.fit(train_X,train_y)
    predict=gboost.predict(test_X)
    gboost_accuracy_score.append(accuracy_score(test_y,predict))
aboost_accuracy_score




gboost=GradientBoostingClassifier(n_estimators=66,random_state=1)    
gboost_model=gboost.fit(train_X,train_y)




predict=gboost_model.predict(test_X)`
predict[0:200,]




ac_score=accuracy_score(test_y,predict)
ac_score




print(classification_report(test_y,predict))




gboost_matrics=confusion_matrix(test_y,predict)
gboost_matrics




sns.heatmap(gboost_matrics,annot=True,cmap='Blues',fmt='g')




probs=gboost_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




gboost_accuracy=accuracy_score(test_y,predict)
gboost_accuracy




gboost_Gini=2*roc_auc-1
gboost_Gini




gboost_AUC=roc_auc
gboost_AUC




R_forest=RandomForestClassifier(n_estimators=50,random_state=1,max_features=15)    
R_forest_model=R_forest.fit(train_X,train_y)
predict=R_forest.predict(test_X)




predict=R_forest_model.predict(test_X)
predict[0:200,]




ac_score=accuracy_score(test_y,predict)
ac_score




print(classification_report(test_y,predict))




R_forest_matrics=confusion_matrix(test_y,predict)
R_forest_matrics




sns.heatmap(R_forest_matrics,annot=True,cmap='Blues',fmt='g')




probs=R_forest_model.predict_proba(test_X)

preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_y, preds)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0,1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




R_forest_accuracy=accuracy_score(test_y,predict)
R_forest_accuracy




R_forest_Gini=2*roc_auc-1
R_forest_Gini




R_forest_AUC=roc_auc
R_forest_AUC




data=[[LR_accuracy,LR_Gini,LR_AUC],[KNN_accuracy,KNN_Gini,KNN_AUC],[NB_accuracy,NB_Gini,NB_AUC],[svc_accuracy,svc_Gini,svc_AUC],
     [dtree_accuracy,dtree_Gini,dtree_AUC],[bagging_accuracy,bagging_Gini,bagging_AUC],[aboost_accuracy,aboost_Gini,aboost_Gini],
     [gboost_accuracy,gboost_Gini,gboost_AUC],[R_forest_accuracy,R_forest_Gini,R_forest_AUC]]




comparison=pd.DataFrame(data,index=['Logestic','KNN','Naive Bayes','SVC','Decision Tree','Bagging','AdaBoosting','GradientBoosting','Random Forest'],columns=['Accuracy','Gini','AUC'])
comparison

