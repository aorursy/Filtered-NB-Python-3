#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
train_df.head()




print (train_df.info())
print ('-'*100)
print (test_df.info())




new_train = train_df
new_test = test_df
from sklearn import cross_validation
x = np.array(new_train.drop(['Survived'],axis = 1))
y = np.array(new_train['Survived'])
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2, random_state = 1)




from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors =5, weights = 'uniform',algorithm= 'kd_tree',leaf_size=50,n_jobs = 1)
clf.fit(x_train,y_train)
acc = clf.score(x_test,y_test)
print ('Accuracy of the model is:',round (acc*100,2), 'Percent')




get_ipython().run_cell_magic('time', '', "var = [10,20,30,40,50,60,70,80,90]\nscore = []\nfor i in var:\n    from sklearn import neighbors\n    nei = KNeighborsClassifier(n_neighbors =5, weights = 'uniform',algorithm= 'kd_tree',leaf_size=50,n_jobs = 1)\n    nei.fit(x_train,y_train)\n    #score.append(nei.score(x_test,y_test))\n    ana = nei.score(x_test,y_test)\n    #print (i,round (ana*100,2))\n    \n#plt.plot(kn,score)\n#plt.show()")




from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=28,n_estimators=10,criterion = 'gini',max_features = 'sqrt')
rf.fit(x_train,y_train)
accrf = rf.score(x_test,y_test)
print ('Accuracy of the model is:',round (accrf*100,2), 'Percent')




var = [1]
for i in var:
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(random_state=28,n_estimators=10,criterion = 'gini',max_features = 'sqrt',min_samples_leaf=i)
    rf.fit(x_train,y_train)
    accrf = rf.score(x_test,y_test)
    if (accrf >=0.84):
        print (i,'Score:',round (accrf*100,2))
    #print (i,'Score:',round (accrf*100,2))




from sklearn.tree import DecisionTreeClassifier
rtree = DecisionTreeClassifier(random_state=1,criterion = 'gini',splitter = 'best',max_features=None,max_depth=3)
rtree.fit(x_train,y_train)
accrf = rtree.score(x_test,y_test)
print ('Accuracy of the model is:',round (accrf*100,2), 'Percent')




var = range(2,100)
for i in var:
    from sklearn.tree import DecisionTreeClassifier
    rf = DecisionTreeClassifier(random_state=1,criterion = 'gini',splitter = 'best',max_features=None,max_depth=3)
    rf.fit(x_train,y_train)
    accrf = rf.score(x_test,y_test)
    #if (accrf >=0.8045):
     #   print (i,'Score:',round (accrf*100,2))
    #print (i,'Score:',round (accrf*100,2))

