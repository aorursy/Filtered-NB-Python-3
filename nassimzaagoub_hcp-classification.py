#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




data = pd.read_json("../input/roam_prescription_based_prediction.jsonl", lines=True)
data.shape




data.head()




Rx = pd.DataFrame([v for v in data["cms_prescription_counts"]])
Rx.shape




Rx.head()




Rx.sum().sort_values(ascending=False).head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Presctiption Counts')




hcp = pd.DataFrame([v for v in data["provider_variables"]])
hcp.shape




hcp.head()




hcp['gender'].value_counts().plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Gender')




hcp['region'].value_counts().plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Region')




hcp['settlement_type'].value_counts().plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Settlement Type')




hcp['specialty'].value_counts().sort_values(ascending=False).head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Provider Specialties')




hcp['years_practicing'].value_counts().plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Years in Practice')




Rx_Humira = Rx[['HUMIRA']]
Rx_Humira.shape




Rx_Humira.head()




clean_data = Rx_Humira.copy()
clean_data['prescribe_label'] = (clean_data['HUMIRA'] > 1)*1
print(clean_data[['HUMIRA', 'prescribe_label']])




y = clean_data[['prescribe_label']].copy()




clean_data['HUMIRA'].head()




y.head()




hcp = pd.DataFrame([v for v in data["provider_variables"]])




get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

lower = hcp['brand_name_rx_count'].mean() - hcp['brand_name_rx_count'].std()
upper = hcp['brand_name_rx_count'].mean() + hcp['brand_name_rx_count'].std()
hist_data = [x for x in hcp[:10000]['brand_name_rx_count'] if x>lower and x<upper ]


# hist_data = hcp['brand_name_rx_count']
print(len(hist_data))




plt.hist(hist_data, 10, normed=False, facecolor='green')

plt.xlabel('Brand Prescriptions')
plt.ylabel('Number of Prescribers')
plt.title('Prescribers Distribution')

plt.grid(True)

plt.show()




specFilter = ['Rheumatology', 'Family', 'Medical', 'Adult Health', 'Procedural Dermatology', 
             'Geriatric Medicine', 'Acute Care', 'MOHS-Micrographic Surgery', 'Allergy & Immunology',
             'Cardiovascular Diseas', 'Clinical & Laboratory', 'Dermatological Immunology', 
             'Surgical Technologist', 'Dermatopathology', 'Hematology & Oncology']

filterMesh = (hcp['specialty'].isin(specFilter)) & (hcp['brand_name_rx_count'] >= 50)
hcp_features = hcp.loc[filterMesh]




hcp_features.shape




hcp_features.columns




X = pd.get_dummies(hcp_features)
X.columns




X.shape




y = pd.merge(y, X, left_index = True, right_index = True)[['prescribe_label']]
y.head()




y.columns




y.shape




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)




hcp_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
hcp_classifier.fit(X_train, y_train)




type(hcp_classifier)




predictions = hcp_classifier.predict(X_test)




predictions[:30]




y_test['prescribe_label'][:20]




accuracy_score(y_true = y_test, y_pred = predictions)




print (hcp_classifier)




from sklearn import metrics
print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))




print("Precision: %0.4f" % metrics.precision_score(y_test, predictions))




print("Recall: %0.4f" % metrics.recall_score(y_test, predictions))




## Get data to plot ROC Curve
fp, tp, th = roc_curve(y_test, predictions)
roc_auc = auc(fp, tp)




## Plot ROC Curve
plt.title('ROC Curve')
plt.plot(fp, tp, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()






