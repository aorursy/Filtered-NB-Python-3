#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install sklearn_crfsuite')
get_ipython().system('pip install eli5')




import nltk
import sklearn_crfsuite
import eli5
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB




import numpy as np # linear algebra
import pandas as pd # data processing

dataset = pd.read_csv("../input/corpus_categorias_treino.csv", index_col=0,)




from sklearn.svm import SVC








qtd_palavras = Pipeline([('qtd_palavras', Qtdpalavras()),('Scaler', MinMaxScaler())])
qtd_palavras.fit_transform(dataset.words) 




dataset.head()




get_ipython().run_line_magic('matplotlib', 'inline')
dataset.category.value_counts().plot(kind='bar')








def avalia_modelo(clf, X, y):
    resultados = cross_val_predict(clf, X, y, cv=5)
    print (pd.crosstab(y, resultados, rownames=['real'], colnames=['Predito'], margins=True))
    return np.mean(cross_val_score(X, y, cv=5))




avalia_modelo(pip_simples,dataset.words,dataset.category)




from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit (1, test_size=0.3, random_state=0)
X = dataset['words'].values
y = dataset['category'].values
a = sss.split(X, y)
for train_index, test_index in a:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]




X_test




# #verificar se proporção de dados do Target manteram a mesma
# #Foi utilizado o trabalho de : https://www.kaggle.com/sudosudoohio/stratified-kfold-xgboost-eda-tutorial-0-281
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
# import seaborn as sns
# sns.set(style='darkgrid')

# #imprimir a distribuição de targets para o dataset todo
# ax = sns.countplot(x = y ,palette="Set2")
# sns.set(font_scale=1.5)
# ax.set_xlabel(' ')
# ax.set_ylabel(' ')
# fig = plt.gcf()
# fig.set_size_inches(10,5)
# ax.set_ylim(top=6000)
# for p in ax.patches:
#     ax.annotate('{:.2f}%'.format(100*p.get_height()/len(y)), (p.get_x()+ 0.3, p.get_height()+10000))

# plt.title('Distribuição das 4 categorias')
# plt.xlabel('Categorias')
# plt.ylabel('Frequência [%]')
# display(plt.show())




# #imprimir a distribuição de targets para o dataset de treino
# targets = y_train
# ax = sns.countplot(x = targets ,palette="Set2")
# sns.set(font_scale=1.5)
# ax.set_xlabel(' ')
# ax.set_ylabel(' ')
# fig = plt.gcf()
# fig.set_size_inches(10,5)
# ax.set_ylim(top=5000)
# for p in ax.patches:
#     ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))
# plt.title('Distribuição de {} dados de treino'.format(X_train.shape))
# plt.xlabel('Reivindicação de seguro de automóvel no próximo ano')
# plt.ylabel('Frequência [%]')
# display(plt.show())
# display(print('{} % de treino'.format(100*X_train.shape[0]/df_train_1.shape[0])))
          
# #imprimir a distribuição de targets para o dataset de teste
# targets = y_test
# ax = sns.countplot(x = targets ,palette="Set2")
# sns.set(font_scale=1.5)
# ax.set_xlabel(' ')
# ax.set_ylabel(' ')
# fig = plt.gcf()
# fig.set_size_inches(10,5)
# ax.set_ylim(top=6000)
# for p in ax.patches:
#     ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x()+ 0.3, p.get_height()+10000))

# plt.title('Distribuição de {} dados de treino'.format(X_train.shape))
# plt.xlabel('Reivindicação de seguro de automóvel no próximo ano')
# plt.ylabel('Frequência [%]')
# display(plt.show())
# display(print('{} % de treino'.format(100*X_test.shape[0]/df_train_1.shape[0])))






