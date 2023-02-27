import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

number = preprocessing.LabelEncoder()

reddit=pd.read_csv("RedditShortDemoSurvey-1-Cleaned.csv")

# def convert(data):
#     number = preprocessing.LabelEncoder()
#     data['Please.indicate.your.gender.'] = number.fit_transform(data.Please.indicate.your.gender.)
#     data['Please.select.the.category.that.includes.your.age.'] = number.fit_transform(data.Please.select.the.category.that.includes.your.age.)
#     data['What.is.your.marital.status.'] = number.fit_transform(data.What.is.your.marital.status.)
#     data['What.best.describes.your.employment.status.'] = number.fit_transform(data.What.best.describes.your.employment.status.)
#     data['Which.one.of.the.following.ranges.includes.your.total.yearly.household.income.before.taxes.'] = number.fit_transform(data.Which.one.of.the.following.ranges.includes.your.total.yearly.household.income.before.taxes.)
#     data=data.fillna(0)
#     return data

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


