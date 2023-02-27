import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

number = preprocessing.LabelEncoder()

reddit=pd.read_csv("RedditShortDemoSurvey-1-Cleaned.csv")


reddit=convert(train)

reddit['is_reddit'] = np.random.uniform(0, 1, len(reddit)) <= .75
reddit, validate = reddit[reddit['is_reddit']==True], reddit[reddit['is_reddit']==False]

lg = LogisticRegression()
lg.fit(x_train, y_train)

Disbursed_lg=lg.predict_proba(x_validate)

fpr, tpr, _ = roc_curve(y_validate, Disbursed_lg[:,1])
roc_auc = auc(fpr, tpr)
print roc_auc

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
disbursed = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, disbursed[:,1])
roc_auc = auc(fpr, tpr)
print roc_auc


