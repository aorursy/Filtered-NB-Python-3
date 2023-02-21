#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import sklearn as sk
from ggplot import *

get_ipython().run_line_magic('matplotlib', 'inline')

data_path = "../input/" # Input data files are available in the "../input/" directory




data = pd.read_csv(data_path + "train.csv", index_col="PassengerId")
print(data.shape)
data.head()




labels = data.Survived

data.drop(["Survived", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
data.head()




print("Missingness Counts")
data.apply(lambda col: (pd.isnull(col).sum()))




# naive imputation: use the mean/mode
data.Age.fillna(data.Age.mean(), inplace=True)
data.Embarked.fillna(data.Embarked.mode().iloc[0], inplace=True)
assert pd.isnull(data).sum().sum() == 0




plots = pd.tools.plotting.scatter_matrix(pd.concat([data, labels], axis=1))
# not sure why 2 sets of plots show up...




pd.concat([data, labels], axis=1).corr()




data.Pclass = data.Pclass.astype(str)
data.Parch = data.Parch.astype(str)

data = pd.get_dummies(data)
data.head()




survival_corr = pd.concat([data, labels], axis=1).corr().Survived.reset_index()
plot = ggplot(data=survival_corr, aesthetics=aes(x='index', weight='Survived')) + geom_bar()        + ggtitle("Correlation with Survival") + ylab("Correlation Coefficient")
#        + theme(axis_text_x=element_text(angle=90, hjust=1))
plot.show()




from sklearn.linear_model import LogisticRegression




model = LogisticRegression()
result = model.fit(data, labels)




result.coef_




np.mean(model.predict(data) == labels)




def preprocess_test_data(data):
    data = data.drop(["Name", "Ticket", "Cabin"], axis=1)
    
    # naive imputation: use the mean/mode
    null_counts = pd.isnull(data).sum()
    for col in null_counts[null_counts > 0].index:
        if type(data[col].iloc[0]) == str:
            data[col] = data[col].fillna(data[col].mode().iloc[0])
        else:
            data[col] = data[col].fillna(data[col].mean())
    
    assert pd.isnull(data).sum().sum() == 0

    data.Pclass = data.Pclass.astype(str)
    data.Parch = data.Parch.astype(str)

    return pd.get_dummies(data)




test = preprocess_test_data(pd.read_csv(data_path + "test.csv", index_col='PassengerId'))
test.drop('Parch_9', axis=1, inplace=True) # not in the training set...
test.head()
# TODO: make sure columns are in the right order (sort alphabetically?)




output = pd.DataFrame({'PassengerId': test.index.values, 'Survived': model.predict(test)})
output.head()




output.to_csv('submission.csv', index=None)






