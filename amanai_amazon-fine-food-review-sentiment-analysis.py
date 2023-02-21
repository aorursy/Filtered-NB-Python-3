#!/usr/bin/env python
# coding: utf-8



# Import Data Set 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

data = pd.read_csv("../input/Reviews.csv")

df= pd.DataFrame(data)
data.head()




df.columns




# Remove reviews of  Rating=3
df1 = df[df.Score != 3]
(df1.shape)




# Converting Rating to Binary class i.e high rating=1, low rating=0
df1['Score']=df1['Score'].apply(lambda x: 1 if (x > 3)  else 0)
df1.head()




# Count Reviews belonged to High Rating and Low Rating
df2= df1.groupby('Score').size()
df2




# Calculate Percentage of High Rating and low Rating
per= df2/sum(df2)*100
per




# plot Histogram between percentage of rating 
df2.plot(kind='bar',title='Label Distribution')
plt.xlabel('rating')
plt.ylabel('values')
# plt.legend()
plt.show()




# Calculate Length of High rating Review V/S Low Level Review
df3=df.iloc[:,[6,8]]
df3 = df[df.Score != 3]
df3['Score']=df3['Score'].apply(lambda x: 1 if (x > 3)  else 0)
df3=df3.iloc[:,[6,8]]
df3['Length'] = df3['Summary'].str.len()
df3=df3.iloc[:,[0,2]]
df3=df3.groupby('Score')['Length'].mean()
df3.head()




# Plot the Graph between Length of High rating Review V/S Low Level Review
df3.plot(kind='bar',color='g',title='positive Reviews have shorter length')
plt.xlabel('rating')
plt.ylabel('avg length of Summary')
# plt.legend()
plt.show()




# Calculate Ratio to find Reviews are Helpfull or not corresponding to Length of the review 
df = df[df.Score != 3]
df['Score']=df['Score'].apply(lambda x: 1 if (x > 3)  else 0)
df['Length'] = df['Summary'].str.len()
df4= df[df.HelpfulnessDenominator!=0]
df4['ratio'] = df4['HelpfulnessNumerator']/df4['HelpfulnessDenominator']
df4['ratio']=df4['ratio'].apply(lambda x: 0 if (x < 0.5)  else 1)
df4=df4.groupby('ratio')['Length'].mean()
df4.head()




# Plot Histogram between Helpfulness ratio and Length of the Summary
df4.plot(kind='bar',color='g',title='longer reviews are helpul')
plt.xlabel('ratio')
plt.ylabel('avg length of Summary')
# plt.legend()
plt.show()




# Calculate Review is high rated and helpul are more than low rated length Reviews
df5= df1[df1.HelpfulnessDenominator!=0]
df5['ratio'] = df5['HelpfulnessNumerator']/df5['HelpfulnessDenominator']
df5['ratio']=df5['ratio'].apply(lambda x: 0 if (x < 0.5)  else 1)
df5['que'] = df5.apply(lambda x : 1 if ((x['Score'] and x['ratio']) ==1) else 0, axis=1)
# print df5.head(3)
df5= df5.groupby('que').size()
df5




# Plot to show that Positive Reviews are Longer in length
df5.plot(kind='bar',color=['g','r'],title='longer reviews are helpul')
plt.xlabel('Rating')
plt.ylabel('Helpullness')
# plt.legend()
plt.show()




#take only three required columns(Score,Summary, Text)
df6=df.iloc[:,[6,8,9]]
df7 = df6[df.Score != 3]
df7['Score']=df7['Score'].apply(lambda x: 1 if (x > 3)  else 0)
df7.Score.value_counts()




# for model building consider 1,2,3 star rating as 0(Low Rating) and 4,5 included as 1(High  Rating)
df8 = data[pd.notnull(data.Summary)]
df8['Score']=df8['Score'].apply(lambda x: 1 if (x > 3)  else 0)
print (df8.shape)
df8.head()




# how to define X and y (from the Review  data) for use with COUNTVECTORIZER
X = df8.Summary+df8.Text
y = df8.Score
print(X.shape)
print(y.shape)




# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




# Create instantiation for CounterVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()




X_train_dtm = vect.fit_transform(X_train)
X_train_dtm
print(type(X_train_dtm))
print(X_train_dtm.shape)
print(X_train_dtm[10,:])




# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm




# Create instantiation of Multinomial Naive bayes and from that check about various parameters
#by default laplace smooting i.e. alpha=1.0
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()




# train the model using X_train_dtm (timing it with an IPython "magic command")
# to know how much time this command will take for execution
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')




# Visulize the effects on term matrix after fitted vocabulary in the model
print(type(X_train_dtm))
print(X_train_dtm.shape)
print(X_test_dtm)




# Predict function is used to predict to which class Test Review Belongs to.
y_pred_class = nb.predict(X_test_dtm)
print(y_pred_class)




# print the confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
gb=metrics.confusion_matrix(y_test, y_pred_class)
print(gb)




#Plot of Confusion Metric
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(gb)
# cm = metrics.confusion_matrix(y_test, y_pred_class, labels=['FAKE', 'REAL'])
pl.title('Confusion matrix')
pl.colorbar()
pl.show()




# calculate accuracy of class predictions
from sklearn import metrics
acc= metrics.accuracy_score(y_test, y_pred_class)
acc




#eqivalent to 1-accuracy
error=1-acc
error




#Recall From above Confusion Metric 
recall=(gb[1,1]+0.0)/sum(gb[1,:])
recall




#precision From above Confusion Metric
pre=(gb[1,1]+0.0)/sum(gb[:,1])
print(pre)




# caculating F1 Score By using HP i.e 
#F1=2*TP/2*TP+FP+FN
F1=(2*pre*recall)/(pre+recall)
F1




# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# Accuracy must be greater than 84% in this case because without any model text belong to positive class
# is 84%, this is imbalanced data set
print('Accuracy', metrics.accuracy_score(y_test, y_pred_class))
print('Recall',metrics.recall_score(y_test,y_pred_class))
print('Precision' ,metrics.precision_score(y_test,y_pred_class))
print('F1-Score',metrics.f1_score(y_test,y_pred_class))




from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_class,target_names=['Negative','Positive']))




# print message text for the false positives (positive review incorrectly classified as negative)
X_test[(y_pred_class==1)&(y_test==0)].count




# calculate predicted probabilities for X_test_dtm 
# We predict the class-membership probability of the samples via the predict_proba method.
y_pred_prob = nb.predict_proba(X_test_dtm)[:,1]
y_pred_prob




# calculate AUC with probabilities values
roc_auc=metrics.roc_auc_score(y_test, y_pred_prob)
roc_auc




# calculate AUC without probabilities values
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_auc)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'g',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




from sklearn.metrics import log_loss
log_error=log_loss(y_test, y_pred_prob)
log_error




from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()




# train the model using X_train_dtm
get_ipython().run_line_magic('time', 'logreg.fit(X_train_dtm, y_train)')




# make class predictions for X_test_dtm
y1_pred_class = logreg.predict(X_test_dtm)
y1_pred_class




# print the confusion matrix
cm= metrics.confusion_matrix(y_test, y1_pred_class)
cm




# Plot confusion Metric
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.show()




# calculate predicted probabilities for X_test_dtm (well calibrated)
y1_pred_prob = logreg.predict_proba(X_test_dtm)[:,1]
y1_pred_prob




# Calculating ROC Rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y1_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)




# PLot AUC For Logistic Regression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_lg)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




# With Probabilities
from sklearn.metrics import log_loss
log_error=log_loss(y_test, y1_pred_prob)
log_error




# calculate accuracy,precision,recall,F1 score
print('Accuracy', metrics.accuracy_score(y_test, y1_pred_class))
print('Recall', metrics.recall_score(y_test,y1_pred_class))
print('Precision', metrics.precision_score(y_test,y1_pred_class))
print('F1 Score', metrics.f1_score(y_test,y1_pred_class))




#  classification_report
print(classification_report(y_test,y1_pred_class,target_names=['Negative','Positive']))




from sklearn.linear_model import LogisticRegression
logreg1 = LogisticRegression(penalty='l1',C=1)
get_ipython().run_line_magic('time', 'logreg1.fit(X_train_dtm, y_train)')




# make class predictions for X_test_dtm
y2_pred_class = logreg1.predict(X_test_dtm)




# print the confusion matrix
cml1= metrics.confusion_matrix(y_test, y2_pred_class)
cml1




# Plot confusion Metric
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(cml1)
pl.title('Confusion matrix')
pl.colorbar()
pl.show()




y2_pred_prob = logreg.predict_proba(X_test_dtm)[:,1]
y2_pred_prob




# Calculating ROC Rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y2_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)




from matplotlib import pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'g',
label='AUC = %0.2f'% roc_lg)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




# Calculating ROC Rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y2_pred_prob)
roc_lg = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_lg)




# calculate accuracy,precision,recall,F1 score
print('Accuracy', metrics.accuracy_score(y_test, y2_pred_class))
print('Recall', metrics.recall_score(y_test,y2_pred_class))
print('Precision', metrics.precision_score(y_test,y2_pred_class))
print('F1 Score', metrics.f1_score(y_test,y2_pred_class))




# With Probabilities
from sklearn.metrics import log_loss
log_error=log_loss(y_test, y2_pred_prob)
log_error




# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names()
len(X_train_tokens)




# examine the first 50 tokens
print(X_train_tokens[0:50])




# Naive Bayes counts the number of times each token appears in each class
# row 1 represents number of times Low rated reviews token appear and row 2corresponds to high rated Reviews tokens
# Leading underscore is Scikit learn convention that the functionality learns while fitting, use by (_)
nb.feature_count_




# rows represent classes, columns represent tokens
nb.feature_count_.shape




# number of times each token appears across all Negative REVIEWS
# here we just slicing above feature count
neg_token_count = nb.feature_count_[0, :]
neg_token_count




# number of times each token appears across all POSITIVE REVIEWS
pos_token_count = nb.feature_count_[1, :]
pos_token_count




# Create a DataFrame of tokens with their separate Negative and Positive Reviews
tokens = pd.DataFrame({'token':X_train_tokens, 'Negative':neg_token_count, 'Positive':pos_token_count}).set_index('token')
tokens.head()




# examine 5 random DataFrame rows
# It shows the number of times a word appear in neg and pos class
tokens.sample(50, random_state=5)




# Naive Bayes counts the number of observations in each class
# it shows that our data is trained on 93426 neg words and 332895 pos word
# In Class_count_  (_)underscore is used because this method is available after model is fitted
nb.class_count_




# add 1 to neg and pos counts to avoid dividing by 0 
tokens['Negative'] = tokens.Negative+ 1
tokens['Positive'] = tokens.Positive + 1
tokens.sample(5, random_state=6)




# convert the negative  and positive counts into frequencies
tokens['Negative']= tokens.Negative / nb.class_count_[0]
tokens['Positive'] = tokens.Positive / nb.class_count_[1]
tokens.sample(5, random_state=6)




# calculate the ratio of Positive-to-Negative for each token
tokens['Positive_ratio'] = tokens.Positive / tokens.Negative
tokens.sample(5, random_state=6)




# examine the DataFrame sorted by Positive_rate
top=tokens.sort_values('Positive_ratio', ascending=False)
print(type(top))
print(top.shape)
print(top.head(100))




# fit train data into Model
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="hinge", penalty="l2")
get_ipython().run_line_magic('time', 'clf.fit(X_train_dtm, y_train)')




# to know everything about instance that we have created
get_ipython().run_line_magic('pinfo', 'clf')




# Visulize the effects on term matrix after fitted vocabulary in the model
print(type(X_train_dtm))
print(X_train_dtm.shape)




# Predict the class label on test class
ys_pred_class = clf.predict(X_test_dtm)
print(ys_pred_class.shape)




# print the confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics
csvm=metrics.confusion_matrix(y_test, ys_pred_class)
csvm




#Plot of Confusion Metric
import matplotlib as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
pl.matshow(csvm)
# cm = metrics.confusion_matrix(y_test, y_pred_class, labels=['FAKE', 'REAL'])
pl.title('Confusion matrix')
pl.colorbar()
pl.show()




print('Accuracy', metrics.accuracy_score(y_test, ys_pred_class))
print('Recall',metrics.recall_score(y_test,ys_pred_class))
print('Precision' ,metrics.precision_score(y_test,ys_pred_class))
print('F1-Score',metrics.f1_score(y_test,ys_pred_class))




from sklearn.metrics import classification_report
print(classification_report(y_test,ys_pred_class,target_names=['Negative','Positive']))




# calculate AUC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, ys_pred_class)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('ROC Rate', roc_auc)




# plot AU-ROC Curve
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'g',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()




from sklearn.metrics import log_loss
log_error=log_loss(y_test, ys_pred_class)
log_error




from sklearn import linear_model, datasets
from sklearn.cross_validation import cross_val_score




from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
Xlg = vect.fit_transform(X)
Xlg




# 3-fold cross-validation with c=100 for logestic regression
logreg = LogisticRegression(C=100)
get_ipython().run_line_magic('time', "scores = cross_val_score(logreg, Xlg, y, cv=3, scoring='accuracy')")
print(scores)




# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())




# 2-fold cross-validation with c=0.05 for logestic regression
logreg = LogisticRegression(penalty='l1',C=0.05)
get_ipython().run_line_magic('time', "scores = cross_val_score(logreg, Xlg, y, cv=2, scoring='accuracy')")
print(scores)




# search for an optimal value of Lambda for Logistic Regression C=1/Lambda
logrg = LogisticRegression(penalty='l1',C=0.05)
L_range = list(range(1,5))
L_scores = []
for l in L_range:
    logreg = LogisticRegression(C=l)
    scores = cross_val_score(logrg, Xlg, y, cv=3, scoring='accuracy')
    L_scores.append(scores.mean())
print(L_scores)




import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the value of Lambda for Logestic Regression(x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(L_range, L_scores)
plt.xlabel('Value of Reverse of regularizer for C ')
plt.ylabel('Cross-Validated Accuracy')




from sklearn.grid_search import GridSearchCV




# define the parameter values that should be searched
l_range = list(range(1,5))
print(l_range)




# create a parameter grid: map the parameter names to the values that should be searched
param_grid=dict(C=l_range)
print(param_grid)




# instantiate the grid
get_ipython().run_line_magic('time', "grid = GridSearchCV(logrg, param_grid, cv=3, scoring='accuracy')")




# fit the grid with data
grid.fit(Xlg, y)




# view the complete results (list of named tuples)
grid.grid_scores_




# create a list of the mean scores only
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)

