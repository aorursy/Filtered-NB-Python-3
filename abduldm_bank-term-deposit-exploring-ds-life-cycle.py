#!/usr/bin/env python
# coding: utf-8



#2.1 Import python libraries to Load Data Set and downloaded to same directory in which this python file saved.
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns




pd.set_option('display.max_columns',500)




#Load Data from csv file
bank_df = pd.read_csv('../input/bank-additional-full.csv', sep = ';')
#bank_df = pd.read_csv('bank-additional-full_v1.csv')
bank_df.info()




bank_df.head(5)




#3.1 Identify Input feature Type - Categorical or Numerical 
#Input Numerical columns (10) - age duration campaign pdays previous emp.var.rate cons.price.idx cons.conf.idx euribor3m nr.employed
#Input Categorical Columns (10) - Job,marital,education,default,housing,loan ,contact,month ,day_of_week,poutcome          

#Check if any missing value in any of the feature
bank_df.isnull().values.any()




#3.2 Convert Target feature value to 0 and 1
bank_df['y'].unique()




bank_df['y'] = np.where(bank_df['y']== 'yes',1,0)




bank_df['y'] = bank_df['y'].astype(np.int64)




# Lets Analyse Categorical input varaible
# For Better Predictive model lets avoid unknown value for marital , job and education. - To DO
# Remove records with value - unknown for education, job and marital and assume it is mandatory for predictive model - To DO
#bank_df = bank_df[(bank_df['marital'] != 'unknown')]
#bank_df = bank_df[(bank_df['education'] != 'unknown')]
#bank_df = bank_df[(bank_df['job'] != 'unknown')]




#Visualize the Categorical feature distribution and relation with target variable. let’s create a User Defined function
def Categorical_Grapgh(data,catfeature,distributionName):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    sns.countplot(x = catfeature, data = data)
    ax.set_xlabel(catfeature, fontsize=15)
    ax.set_ylabel('Count', fontsize=15)
    ax.set_title(distributionName, fontsize=15)
    ax.tick_params(labelsize=15)
    sns.despine()




Categorical_Grapgh(bank_df,'job','job Distribution')
Categorical_Grapgh(bank_df,'marital','maritial Distribution')
Categorical_Grapgh(bank_df,'education','education Distribution')
Categorical_Grapgh(bank_df,'default','default Distribution')
Categorical_Grapgh(bank_df,'housing','housing Distribution')
Categorical_Grapgh(bank_df,'loan','loan Distribution')
Categorical_Grapgh(bank_df,'contact','contact Distribution')
Categorical_Grapgh(bank_df,'month','month Distribution')
Categorical_Grapgh(bank_df,'day_of_week','day_of_week Distribution')
Categorical_Grapgh(bank_df,'poutcome','poutcome Distribution')




#Check How Categorize variables correlated with Target Variables and How it impacted.
from scipy import stats




#Check How Job Type , Education are correlated with Target Variable
bank_df.groupby(['job','y']).y.count()
#Admin are more interested in Term Deposit.




F, p = stats.f_oneway(bank_df[bank_df.job=='admin.'].y,
                      bank_df[bank_df.job=='blue-collar'].y,
                      bank_df[bank_df.job=='entrepreneur'].y,
                      bank_df[bank_df.job=='housemaid'].y,
                      bank_df[bank_df.job=='management'].y,
                      bank_df[bank_df.job=='retired'].y,
                      bank_df[bank_df.job=='self-employed'].y,
                      bank_df[bank_df.job=='services'].y,
                      bank_df[bank_df.job=='student'].y,
                      bank_df[bank_df.job=='technician'].y,
                      bank_df[bank_df.job=='unemployed'].y
                      )
print(F)




# Seems JOB has little impact on target variable




bank_df.groupby(['marital','y']).y.count()
#married people are more interested in Term Deposit




bank_df.groupby(['job','marital','y']).y.count()
# And Admin - married people are more interested in Term Deposit.




bank_df.groupby(['contact','y']).y.count()
#Contact field has good correlation with Target variable. Since we have two observation for contact lets convert this to binary format. cellular -1 and telephone=0




F, p = stats.f_oneway(bank_df[bank_df.contact=='telephone'].y,
                      bank_df[bank_df.contact=='cellular'].y)
print(F)




F, p = stats.f_oneway(bank_df[bank_df.day_of_week=='mon'].y,
                      bank_df[bank_df.day_of_week=='tue'].y,
                      bank_df[bank_df.day_of_week=='wed'].y,
                      bank_df[bank_df.day_of_week=='thu'].y,
                      bank_df[bank_df.day_of_week=='fri'].y)
print(F)




bank_df.groupby(['day_of_week','y']).age.count()




# day_of_week - No significant correlation with Target variable




F, p = stats.f_oneway(bank_df[bank_df.loan=='no'].y,
                      bank_df[bank_df.loan=='yes'].y,
                      bank_df[bank_df.loan=='unknown'].y)
print(F)




# Loan - No correlation with Target variable




bank_df.groupby(['loan','y']).age.count()




bank_df.groupby(['default','y']).age.count()




F, p = stats.f_oneway(bank_df[bank_df.default=='no'].y,
                      bank_df[bank_df.default=='yes'].y,
                      bank_df[bank_df.default=='unknown'].y)
print(F)




bank_df.groupby(['housing','y']).age.count()




F, p = stats.f_oneway(bank_df[bank_df.housing=='no'].y,
                      bank_df[bank_df.housing=='yes'].y,
                      bank_df[bank_df.housing=='unknown'].y)
print(F)




# housing - No significant Relation with Target Variable




bank_df.groupby(['poutcome','y']).age.count()




F, p = stats.f_oneway(bank_df[bank_df.poutcome=='success'].y,
                      bank_df[bank_df.poutcome=='failure'].y,
                      bank_df[bank_df.poutcome=='nonexistent'].y)
print(F)




# poutcome - Good Relation with Target Variable




bank_df.groupby(['month','y']).age.count()




#Convert Categorical column to Continues Type. use Label conceding for ordinal category and one hot encoding for Nominal
print(bank_df.job.unique())
#job  - Nominal
print(bank_df.marital.unique())
#Maritial Nominal
print(bank_df.education.unique())
#education - Ordinary
print(bank_df.default.unique())
# seems Ordinary if we put -1 for unknown
print('housing', bank_df.housing.unique())
# seems Ordinary if we put -1 for unknown
print(bank_df.loan.unique())
# seems Ordinary if we put -1 for unknown
print(bank_df.contact.unique())
# Nominal
print(bank_df.month.unique())
#ordinal
print(bank_df.day_of_week.unique())
#ordinal
print(bank_df.poutcome.unique())
#ordinal if we put -1 for nonexistent




from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

bank_df = pd.get_dummies(bank_df,columns=['job','marital','education','default','housing','loan'])
bank_df['month'].replace(['mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], [1,2,3,4,5,6,7,8,9,10], inplace  = True)
#labelencoder_X.fit(bank_df['day_of_week'])
#bank_df['day_of_week'] = labelencoder_X.transform(bank_df['day_of_week'])
bank_df['day_of_week'].replace(['mon','tue','wed','thu','fri'],[1,2,3,4,5],inplace=True)
#labelencoder_X.fit(bank_df['poutcome'])
#bank_df['poutcome'] = labelencoder_X.transform(bank_df['poutcome'])
bank_df['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)
bank_df['isCellular'] = bank_df['contact']
bank_df['isCellular'].replace(['telephone', 'cellular'], [0,1], inplace  = True)
#bank_df['default'].replace(['unknown','no', 'yes'], [1,2,3], inplace  = True)
#bank_df['housing'].replace(['no' ,'yes'], [0,1], inplace  = True)




#bank_df = bank_df.drop(['loan'],axis=1)
#bank_df = bank_df.drop(['day_of_week'],axis=1)
#bank_df = bank_df.drop(['housing'],axis=1)
bank_df = bank_df.drop(['contact'],axis=1)




#Lets Analyze Continuous features. - Use Describe and Correlation function.
bank_df.describe()




bank_df.corr()
# Input feature - nr.employed and  euribor3m (.94) and emp.var.rate and nr.employed (.90) 
#and euribor3m and emp.var.rate (.97) are more correlated and we can remove on column.
# And lets Remove columns - euribor3m and emp.var.rate 




plt.figure(figsize=(40,40)) 
sns.heatmap(bank_df.corr())




#Remove Low Correlated input variable 
bank_df = bank_df.drop(['euribor3m'],axis=1)
bank_df = bank_df.drop(['emp.var.rate'],axis=1)




# Check outlier if any for Numberic column.
bank_df.age.plot(kind='box')
# There are outlier and check max age and age greated than 90




print(bank_df.age.max())
bank_df[bank_df['age'] > 80].head(100)




bank_df.age.plot(kind='hist')
# it is bit positively skewed but it is ok and seems no high dependency with Output variable




bank_df.age.plot(kind='kde')




# Create Binning for all numeric fields base on Box plot quantile
def binning(dataframe,featureName):
    print (featureName)
    q1 = dataframe[featureName].quantile(0.25)
    q2 = dataframe[featureName].quantile(0.50)
    q3 = dataframe[featureName].quantile(0.75)
    dataframe.loc[(dataframe[featureName] <= q1), featureName] = 1
    dataframe.loc[(dataframe[featureName] > q1) & (dataframe[featureName] <= q2), featureName] = 2
    dataframe.loc[(dataframe[featureName] > q2) & (dataframe[featureName] <= q3), featureName] = 3
    dataframe.loc[(dataframe[featureName] > q3), featureName] = 4 
    print (q1, q2, q3)
    




binning(bank_df,'age')




# let check campaign field now and it is positively skewed..
bank_df.campaign.plot(kind='hist')




bank_df.campaign.plot(kind='box')
# lot of exreme values.




print(bank_df.campaign.max())
print(bank_df.campaign.mean())
print(bank_df.campaign.median())
print(bank_df.campaign.unique())
print('Y=1 for campaign > 10' , bank_df[(bank_df['campaign'] > 10) & (bank_df['y'] ==1)].age.count())
print('Y=1 for campaign < 10' , bank_df[(bank_df['campaign'] <= 10) & (bank_df['y'] ==1)].age.count())
print('Y=1 for campaign = 1' , bank_df[(bank_df['campaign'] == 1) & (bank_df['y'] ==1)].age.count())




bank_df.groupby(['campaign','y']).y.count()




bank_df['campaign'].describe()




q1 = bank_df['campaign'].quantile(0.25)
q2 = bank_df['campaign'].quantile(0.50)
q3 = bank_df['campaign'].quantile(0.75)

print(q1)
print(q2)
print(q3)

iqr = q3-q1 #Interquartile range

extreme_low_campaign = q1-1.5*iqr
extreme_high_capmaign = q3+1.5*iqr

print (extreme_low_campaign)
print (extreme_high_capmaign)




binning(bank_df,'campaign')




# pdays - number of days that passed by after the client was last contacted from a previous 
#campaign (numeric; 999 means client was not previously contacted)
# exclude Pdays = 999
bank_df[bank_df['pdays'] != 999].pdays.plot(kind='hist')




bank_df[bank_df['pdays'] != 999].pdays.plot(kind='box')
#sems Box plot is not applicable here 




# lets replace 999 with 0 to avoid extrem upper bound impact in machine learning.

bank_df.loc[(bank_df['pdays'] >= 0) & (bank_df['pdays'] <= 5), 'pdays'] = 2
bank_df.loc[(bank_df['pdays'] > 5) & (bank_df['pdays'] <= 10), 'pdays'] = 3
bank_df.loc[(bank_df['pdays'] > 10) & (bank_df['pdays'] <= 20), 'pdays'] = 4
bank_df.loc[(bank_df['pdays'] > 20) & (bank_df['pdays'] != 999) , 'pdays'] = 5 

bank_df.loc[(bank_df['pdays'] == 999), 'pdays'] = 1




bank_df.pdays.unique()




bank_df.groupby(['previous','y']).age.count()




bank_df.duration.plot(kind='box')

#sems Box plot is not applicable here 




bank_df.duration.plot(kind='hist')




bank_df.duration.describe()




bank_df[bank_df['duration'] > 3000]




bank_df




#binning(bank_df,'duration')




bank_df[['cons.price.idx','cons.conf.idx','nr.employed','y']].describe()




bank_df[['cons.price.idx','cons.conf.idx','nr.employed','y']].corr()




from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV




#UDF for create model
m_bank_df=bank_df




m_bank_df.columns




def Split_Data(processeddata):
    #processeddata['y'] = np.where(processeddata['y'] == 'yes',1,0)
    columns = [column for column in processeddata.columns if column != 'y']
    columns  = ['y']+columns
    processeddata= processeddata[columns]

    y=processeddata['y'].ravel()
    del processeddata['y']
    X= processeddata.as_matrix().astype('float')

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=0)
    return X_train,X_test,y_train,y_test




def Convert_Model(X_train,y_train,X_test,y_test,classifier):
     from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix
     classifier.fit(X_train,y_train) 
     print(classifier.score(X_test,y_test)) 
     print(confusion_matrix(y_test,classifier.predict(X_test)))
     print(accuracy_score(y_test,classifier.predict(X_test)))
     print(precision_score(y_test,classifier.predict(X_test)))
     print(recall_score(y_test,classifier.predict(X_test)))
     f1 = 2 * precision_score(y_test,classifier.predict(X_test)) * recall_score(y_test,classifier.predict(X_test)) / (precision_score(y_test,classifier.predict(X_test)) + recall_score(y_test,classifier.predict(X_test)))
     print("f1 score", f1)
     return classifier




X_train,X_test,y_train,y_test = Split_Data(m_bank_df)




# inport Dummy Classifier for creating Base Model
from sklearn.dummy import DummyClassifier
classifier = DummyClassifier(strategy='most_frequent',random_state=0)
finalModel = Convert_Model(X_train,y_train,X_test,y_test,classifier)




from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




# inport Dummy Classifier for creating Base Model
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
finalModel_lr = Convert_Model(X_train,y_train,X_test,y_test,classifier_lr)




# roc curve and auc on imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = finalModel_lr.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()




from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
finalModel_gb = Convert_Model(X_train,y_train,X_test,y_test,gb)




# roc curve and auc on imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
probs = finalModel_gb.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the precision-recall curve for the model
pyplot.plot(fpr, tpr, marker='.')
# show the plot
pyplot.show()




#Ignore Duration field as it is - Duration: last contact duration, in seconds (numeric). 
#Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
m_bank_df = m_bank_df.drop(['duration'],axis=1)




X_train,X_test,y_train,y_test = Split_Data(m_bank_df)




from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
finalModel_gb = Convert_Model(X_train,y_train,X_test,y_test,gb)




# import pickle library
import pickle




# create the file paths
model_file_path = os.path.join('gb_model.pkl')
#scaler_file_path = os.path.join(os.path.pardir,'models','lr_scaler.pkl')




# open the files to write 
model_file_pickle = open(model_file_path, 'wb')
#scaler_file_pickle = open(scaler_file_path, 'wb')




# open the files to write 
model_file_pickle = open(model_file_path, 'wb')
#scaler_file_pickle = open(scaler_file_path, 'wb')




# persist the model and scaler
pickle.dump(finalModel_lr, model_file_pickle)




# close the file
model_file_pickle.close()




import os
hello_world_script_file = os.path.join('bank_api.py')




get_ipython().run_cell_magic('writefile', '$hello_world_script_file', 'import pandas as pd\nimport json\nfrom flask import Flask, request\nimport numpy as np\nimport os\nimport pickle\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split, GridSearchCV\n\napp = Flask(__name__)\n\n# Load Model and Scaler Files\nmodel_path = os.path.join(\'models\')\nmodel_filepath = os.path.join(model_path, \'lr_model.pkl\')\n\nmodel = pickle.load(open(model_filepath))\n\n\n@app.route(\'/api\', methods=[\'POST\'])\ndef say_hello():\n    print(\'Reached here  1\')\n    data = json.dumps(request.get_json(force=True))\n    print(data)\n    print(\'Reached here  2 \')\n      \n    # create pandas dataframe using json string\n    my_df = pd.read_json(data)\n   \n    print(my_df[\'age\'])\n   \n    my_df.info()\n    #my_df=json.loads(request.get_json(force=True))\n    print(my_df[\'campaign\'])\n    my_df[\'age\'] = my_df[\'age\'].astype(np.int64)\n    my_df[\'campaign\'] = my_df[\'campaign\'].astype(np.int64)\n    my_df[\'pdays\'] = my_df[\'pdays\'].astype(np.int64)\n    my_df[\'previous\'] = my_df[\'previous\'].astype(np.int64)\n    \n    inputData=my_df\n    print("Data Preparation and input structure")\n    inputData = inputData.drop([\'euribor3m\'],axis=1)\n    inputData = inputData.drop([\'loan\'],axis=1)\n    inputData = inputData.drop([\'day_of_week\'],axis=1)\n    inputData = inputData.drop([\'emp.var.rate\'],axis=1)\n    inputData = inputData.drop([\'duration\'],axis=1)\n    inputData = inputData.drop([\'housing\'],axis=1)\n    #inputData = inputData.drop([\'cons.conf.idx\'],axis=1)\n   \n    inputData = pd.get_dummies(inputData,columns=[\'job\',\'marital\',\'education\',\'default\'])\n    inputData[\'month\'].replace([\'mar\',\'apr\',\'may\',\'jun\',\'jul\',\'aug\',\'sep\',\'oct\',\'nov\',\'dec\'], [1,2,3,4,5,6,7,8,9,10], inplace  = True)\n    inputData[\'poutcome\'].replace([\'nonexistent\', \'failure\', \'success\'], [1,2,3], inplace  = True)\n    inputData[\'isCellular\'] =inputData[\'contact\']\n    inputData[\'isCellular\'].replace([\'telephone\', \'cellular\'], [0,1], inplace  = True)\n    inputData = inputData.drop([\'contact\'],axis=1)\n  \n    print("Data Preparation and input structure")\n    \n    inputData.loc[(inputData[\'age\'] <=32),\'age\'] = 1\n    inputData.loc[(inputData[\'age\'] > 32) &  (inputData[\'age\'] <= 38),  \'age\'] = 2\n    inputData.loc[(inputData[\'age\'] > 38) & (inputData[\'age\'] <= 47),  \'age\'] = 3\n    inputData.loc[(inputData[\'age\'] > 47), \'age\'] = 4 \n\n    inputData.loc[(inputData[\'campaign\'] <= 1) , \'campaign\'] = 1\n    inputData.loc[(inputData[\'campaign\'] > 1) & (inputData[\'campaign\'] <= 2), \'campaign\'] = 2\n    inputData.loc[(inputData[\'campaign\'] > 2) & (inputData[\'campaign\'] <= 3), \'campaign\'] = 3\n    inputData.loc[(inputData[\'campaign\'] > 3) , \'campaign\'] = 4 \n \n    inputData.loc[(inputData[\'pdays\'] >= 0) & (inputData[\'pdays\'] <= 5), \'pdays\'] = 2\n    inputData.loc[(inputData[\'pdays\'] > 5) & (inputData[\'pdays\'] <= 10), \'pdays\'] = 3\n    inputData.loc[(inputData[\'pdays\'] > 10) & (inputData[\'pdays\'] <= 20), \'pdays\'] = 4\n    inputData.loc[(inputData[\'pdays\'] > 20) & (inputData[\'pdays\'] != 999) , \'pdays\'] = 5 \n    inputData.loc[(inputData[\'pdays\'] == 999), \'pdays\'] = 1\n    \n    column = [u\'age\', u\'contact\', u\'month\', u\'campaign\', u\'pdays\', u\'previous\',\n       u\'poutcome\', u\'cons.price.idx\', u\'nr.employed\', u\'y\', u\'job_admin.\',\n       u\'job_blue-collar\', u\'job_entrepreneur\', u\'job_housemaid\',\n       u\'job_management\', u\'job_retired\', u\'job_self-employed\',\n       u\'job_services\', u\'job_student\', u\'job_technician\', u\'job_unemployed\',\n       u\'job_unknown\', u\'marital_divorced\', u\'marital_married\',\n       u\'marital_single\', u\'marital_unknown\', u\'education_basic.4y\',\n       u\'education_basic.6y\', u\'education_basic.9y\', u\'education_high.school\',\n       u\'education_illiterate\', u\'education_professional.course\',\n       u\'education_university.degree\', u\'education_unknown\', u\'housing_no\',\n       u\'housing_unknown\', u\'housing_yes\', u\'default_no\', u\'default_unknown\',\n       u\'default_yes\']\n\n    input_columns = inputData.columns\n    dif = list(set(column) - set(input_columns))\n    print (dif)\n\n    for x in dif:\n        if x != \'y\':\n            inputData.insert(1, x, 0)\n            inputData[x] = inputData[x].astype(np.uint8)\n        \n    print(list(set(column) - set(input_columns)))\n   \n    inputData.info()\n    prediction = model.predict(inputData)\n    result = pd.DataFrame({\'result\': prediction})\n    print (result[\'result\'])\n    return result.to_json(orient=\'records\')\n    #return "{0}".format(result[\'result\'])\n\nif __name__ == \'__main__\':\n    app.run(port=10001, debug=True)\n   ')

