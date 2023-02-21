#!/usr/bin/env python
# coding: utf-8



### Imports

## Ignorando warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)

## Blibliotecas Basicas
import operator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from collections import Counter
from matplotlib import pyplot as plt

## Bibliotecas de Processamento e Analise
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

## Bibliotecas de Machine Learning - 14 Alg
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier




arquivo_treino = '../input/dataset_treino.csv'
arquivo_teste = '../input/dataset_teste.csv'

df_treino = pd.read_csv(arquivo_treino)
df_teste = pd.read_csv(arquivo_teste)




Y = df_treino['classe']
X = df_treino.drop(['classe'], axis= 1)

colunas = X.columns




df_treino.head(10)




print("O Conjunto de dados de treino possui as dimensões: %d Linhas x %d Colunas" % (df_treino.shape))
print("O Conjunto de dados de teste possui as dimensões: %d Linhas x %d Colunas" % (df_teste.shape))




df_treino.dtypes




df_treino.describe()




fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(1, len(colunas)):  
    n = df_treino.groupby(colunas[i])[colunas[i]].count().sort_values(ascending=False)
    
    ax = fig.add_subplot(3, 3, i)
    descending_order = n.index
    ax = fig.add_subplot(sns.countplot(y=df_treino[colunas[i]],order=descending_order))

    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    ax.plot()




sns.set(rc={'figure.figsize':(12.7,9.27)})
sns.heatmap(df_treino.corr(), annot=True)




df_treino.hist()
plt.show()




df_treino.plot(kind='density', subplots=True, layout=(3,4), sharex= False)
plt.show()




sns.set(rc={'figure.figsize':(11.7,9.27)})
sns.boxplot(data=df_treino, orient='v')




def normalizar_MinMaxScaler(X):
    colunas_normalizadas = list(X.columns)
    x_scaler = MinMaxScaler().fit_transform(X)
    X_norm_mms = pd.DataFrame(x_scaler, columns = colunas_normalizadas)
    return X_norm_mms




def normalizar_StandardScaler(X):
    colunas_normalizadas = list(X.columns)
    x_scaler = StandardScaler().fit_transform(X)
    X_norm_ss = pd.DataFrame(x_scaler, columns = colunas_normalizadas)
    return X_norm_ss




def FeatureSelection(X,Y):
    modelo = ExtraTreesClassifier()
    modelo.fit(X,Y)
    
    pontuacao = list(modelo.feature_importances_)  
    
    index = np.arange(len(colunas))
    plt.barh(index, pontuacao)
    plt.xlabel('Feature', fontsize=11)
    plt.ylabel('Pontuacao', fontsize=11)
    plt.yticks(index, colunas, fontsize=10, rotation=30)
    plt.title('Pontuacao das variaveis para possiveis variaveis features.')
    plt.show()
    
    print("As Melhores Features São:\n")
    melhores_colunas = []
    for i,j in zip(list(colunas), pontuacao):
        if(j > 0.1):
            print(" Coluna: %s: %s " % (i,j))
            melhores_colunas.append(i)
     
    print("\nO Modelo com as melhores Features Ficou:\n")
    print(X[melhores_colunas].head(5))
    return(X[melhores_colunas])




def busca_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return IQR
        
def remover_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = busca_outliers_iqr(df)
    qtd_shape_original = df.shape[0]
    df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print("Dataset original: %s" % qtd_shape_original)
    print("Dataset sem outliers: %s" % df.shape[0])
    print("Quantidade de outliers removidos: %s" % (qtd_shape_original - df.shape[0]))
    sns.set(rc={'figure.figsize':(11.7,9.27)})
    sns.boxplot(data=df, orient='v')
    return df




def busca_outliers_zscore(df):
    z = np.abs(stats.zscore(df))
    return z

def remover_outliers_zscore(df):
    z = busca_outliers_zscore(df)
    qtd_shape_original = df.shape[0]
    df = df[(z < 3).all(axis=1)]
    print("Dataset original: %s" % qtd_shape_original)
    print("Dataset sem outliers: %s" % df.shape[0])
    print("Quantidade de outliers removidos: %s" % (qtd_shape_original - df.shape[0]))
    sns.set(rc={'figure.figsize':(11.7,9.27)})
    sns.boxplot(data=df, orient='v')
    return df




def ranking(X,Y):
    modelos = []
    modelos.append(('SVM', SVC()))
    modelos.append(('XGB', XGBClassifier()))
    modelos.append(('NB', GaussianNB()))   
    modelos.append(('CART', DecisionTreeClassifier()))
    modelos.append(('MLP', MLPClassifier()))
    modelos.append(('KNN', KNeighborsClassifier()))      
    modelos.append(('LR', LogisticRegression()))
    modelos.append(('SGD', SGDClassifier()))    
    modelos.append(('LDA', LinearDiscriminantAnalysis()))   
    modelos.append(('ETC', ExtraTreesClassifier()))
    modelos.append(('BGC', BaggingClassifier()))
    modelos.append(('GBC', GradientBoostingClassifier()))   
    modelos.append(('RFC', RandomForestClassifier()))
    modelos.append(('ADA', AdaBoostClassifier()))
     
    resultados = []
    nomes = []
    _sup = 0
    msg = ""
    for nome, modelo in modelos:        
        cv_results = cross_val_score(modelo, X, Y, cv=5, scoring='accuracy')
        resultados.append(cv_results)
        nomes.append(nome)
        _sup += 1
        msg += "%s: %f (%f)  -  " % (nome, cv_results.mean(), cv_results.std())
        if(_sup == 4):            
            print(msg)
            msg = ""
            _sup = 0
        
    _max = []
    for i,j in zip(resultados, nomes):
        _max.append((i.mean(), j))        
    print("\n\tA melhor pontuação foi de: %f para o algoritmo %s " % max(_max))
             
    sns.set(rc={'figure.figsize':(9.7,7.27)})
    fig = sns.boxplot(data=resultados, whis=np.inf, width=.35)
    plt.title("Comparacao de Algoritmos")
    y_pos = np.arange(len(nomes))
    plt.xticks(y_pos, nomes)
    plt.show(fig) 
    return(max(_max))




X_norm_mms = normalizar_MinMaxScaler(X)
X_norm_mms.head(3)




X_norm_ss = normalizar_StandardScaler(X)
X_norm_ss.head(3)




X_fs = FeatureSelection(X,Y)




X_norm_mms_fs = normalizar_MinMaxScaler(X_fs)
X_norm_mms_fs.head(3)




X_norm_ss_fs = normalizar_StandardScaler(X_fs)
X_norm_ss_fs.head(3)




df_iqr = remover_outliers_iqr(df_treino)
print(df_iqr.shape)

X_iqr = df_iqr.drop('classe', axis=1)
Y_iqr = df_iqr.classe




df_zscore = remover_outliers_zscore(df_treino)
print(df_zscore.shape)

X_zscore = df_zscore.drop('classe', axis=1)
Y_zscore = df_zscore.classe




X_iqr_norm_mms = normalizar_MinMaxScaler(X_iqr)
print(X_iqr_norm_mms.shape)
X_iqr_norm_mms.head(3)




X_zscore_norm_mms = normalizar_MinMaxScaler(X_zscore)
print(X_zscore_norm_mms.shape)
X_zscore_norm_mms.head(3)




X_iqr_norm_ss = normalizar_StandardScaler(X_iqr)
print(X_iqr_norm_ss.shape)
X_iqr_norm_ss.head(3)




X_zscore_norm_ss = normalizar_StandardScaler(X_zscore)
print(X_zscore_norm_ss.shape)
X_zscore_norm_ss.head(3)




X_iqr_fs = FeatureSelection(X_iqr, Y_iqr)
X_iqr_fs_norm_mms = normalizar_MinMaxScaler(X_iqr_fs)
X_iqr_fs_norm_mms.head(3)




X_zscore_fs = FeatureSelection(X_zscore, Y_zscore)
X_zscore_fs_norm_mms = normalizar_MinMaxScaler(X_zscore_fs)
X_zscore_fs_norm_mms.head(3)




X_iqr_fs_norm_ss = normalizar_StandardScaler(X_iqr_fs)
X_iqr_fs_norm_ss.head(3)




X_zscore_fs_norm_ss = normalizar_StandardScaler(X_zscore_fs)
X_zscore_fs_norm_ss.head(3)




datasets = []

datasets.append((X, Y, "Padrao"))
datasets.append((X_norm_mms, Y, "MinMaxScaler"))
datasets.append((X_norm_ss, Y, "StandardScaler"))
datasets.append((X_fs, Y, "FeatureSelection"))
datasets.append((X_norm_mms_fs, Y, "MinMaxScaler_e_FeatureSelection"))
datasets.append((X_norm_ss_fs, Y, "StandardScaler_e_FeatureSelection"))
datasets.append((X_zscore, Y_zscore, "Z-Score"))
datasets.append((X_iqr, Y_iqr, "IQR"))
datasets.append((X_zscore_norm_mms, Y_zscore, "MinMaxScaler_e_Z-Score"))
datasets.append((X_zscore_norm_ss, Y_zscore, "StandardScaler_e_Z-Score"))
datasets.append((X_iqr_norm_mms, Y_iqr, "MinMaxScaler_e_IQR"))
datasets.append((X_iqr_norm_ss, Y_iqr, "StandardScaler_e_IQR"))
datasets.append((X_zscore_fs, Y_zscore, "FeatureSelection_e_Z-Score"))
datasets.append((X_iqr_fs, Y_iqr, "FeatureSelection_e_IQR"))
datasets.append((X_iqr_fs_norm_mms, Y_iqr, "MinMaxScaler_FeatureSelection_e_IQR"))
datasets.append((X_iqr_fs_norm_ss,Y_iqr, "StandardScaler_FeatureSelection_e_IQR"))
datasets.append((X_zscore_fs_norm_mms, Y_zscore, "MinMaxScaler_FeatureSelection_e_Z-Score"))
datasets.append((X_zscore_fs_norm_ss, Y_zscore, "StandardScaler_FeatureSelection_e_Z-Score"))

def print_shape(datasets):
    for (X, Y, descricao) in datasets:
        print("O Dataset [%s] contem o shape: \n\t%d Colunas e %d Linhas, Tamanho de X: %d\n" 
              % (descricao, len(X.columns), len(Y), len(X)))
        
print_shape(datasets)




def scores_x_datasets(datasets):
    scores = []
    for (X, Y, descricao) in datasets:
        scores.append((ranking(X,Y), descricao))
        
    return scores

scores = scores_x_datasets(datasets)
sorted(scores, key = operator.itemgetter(0), reverse=True)




def top_algoritmos(scores):
    pontuacao = []
    algoritmo = []
    dataset = []
    scores = sorted(scores, key = operator.itemgetter(0), reverse=True)
    for i in scores:
        _str = str(i)       
        
        _pontuacao = float(_str.split()[0].replace("(", ""). replace(",", ""))
        pontuacao.append(_pontuacao)
        
        _algoritmo = _str.split()[1].replace("'", "").replace(")", "").replace(",", "")
        algoritmo.append(_algoritmo)
        
        _dataset = _str.split()[2].replace("'", "").replace(")", "")
        dataset.append(_dataset)
        
    return Counter(algoritmo)
        
top_algoritmos(scores)




def grid_search(datasets, algoritmos):
    
    scores = []
    for algoritmo in algoritmos:        
        for (X, Y, alg_desc) in datasets:

            clf = None
            params = {}

            if(algoritmo == 'SVM'):
                params = {
                    'kernel':('linear', 'rbf'), 
                    'C':(1,0.25,0.5,0.75),
                    'gamma': (1,2,3,'auto'),
                    'decision_function_shape':('ovo','ovr'),
                    'shrinking':(True,False)
                }
                clf = SVC()
            elif(algoritmo == 'XGB'):
                params = {
                    'min_child_weight': [1, 5, 10],
                    'gamma': [0.5, 1, 1.5, 2, 5],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'max_depth': [3, 4, 5]
                }
                clf = XGBClassifier()
            elif(algoritmo == 'NB'):
                params = { }
                clf = GaussianNB()
            elif(algoritmo == 'CART'):
                params = {
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': [2, 10, 20],
                    'max_depth': [None, 2, 5, 10],
                    'min_samples_leaf': [1, 5, 10],
                    'max_leaf_nodes': [None, 5, 10, 20],
                }
                clf = DecisionTreeClassifier()
            elif(algoritmo == 'MLP'):
                params = {
                    'solver': ['lbfgs'],
                    'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 
                    'alpha': 10.0 ** -np.arange(1, 10), 
                    'hidden_layer_sizes':np.arange(10, 15), 
                    'random_state':[0,1,2,3,4,5,6,7,8,9]
                }
                clf = MLPClassifier()
            elif(algoritmo == 'KNN'):
                params = {
                    'n_neighbors':np.arange(1,50)
                }
                clf = KNeighborsClassifier()
            elif(algoritmo == 'LR'):
                params = {
                    'C':np.logspace(-3,3,7), 
                    'penalty':['l1','l2']
                }
                clf = LogisticRegression()
            elif(algoritmo == 'SGD'):
                params = {
                    'loss' : ['hinge', 'log', 'squared_hinge', 'modified_huber'],
                    'alpha' : [0.0001, 0.001, 0.01, 0.1],
                    'penalty' : ['l2', 'l1', 'none'],
                }
                clf = SGDClassifier()
            elif(algoritmo == 'LDA'):
                params = { }
                clf = LinearDiscriminantAnalysis()
            elif(algoritmo == 'ETC'):
                params = {
                    'n_estimators': np.arange(1, 200)
                }
                clf = ExtraTreesClassifier()
            elif(algoritmo == 'BGC'):
                params = {
                    'base_estimator__max_depth': [3,5,10,20],
                    'base_estimator__max_features': [None, 'auto'],
                    'base_estimator__min_samples_leaf': [1, 3, 5, 7, 10],
                    'base_estimator__min_samples_split': [2, 5, 7],
                    'bootstrap_features': [False, True],
                    'max_features': [0.5, 0.7, 1.0],
                    'max_samples': [0.5, 0.7, 1.0],
                    'n_estimators': [2, 5, 10, 20],
                }
                clf = BaggingClassifier()
            elif(algoritmo == 'GBC'):
                params = {
                    'loss': ['deviance'],
                    'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                    'min_samples_split': np.linspace(0.1, 0.5, 12),
                    'min_samples_leaf': np.linspace(0.1, 0.5, 12),
                    'max_depth': [3,5,8],
                    'max_features':['log2','sqrt'],
                    'criterion': ['friedman_mse',  'mae'],
                    'subsample': [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                    'n_estimators' :[10]
                }
                clf = GradientBoostingClassifier()
            elif(algoritmo == 'RFC'):
                params = {
                    'n_estimators': [200, 500],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth' : [4,5,6,7,8],
                    'criterion' : ['gini', 'entropy']
                }
                clf = RandomForestClassifier()
            elif(algoritmo == 'ADA'):
                params = {
                    'base_estimator__criterion' : ['gini', 'entropy'],
                    'base_estimator__splitter' : ['best', 'random'],
                    'n_estimators': [1, 2]
                }
                clf = AdaBoostClassifier()


            grid = GridSearchCV(clf, params, cv=5, n_jobs=-1)
            grid.fit(X,Y)
            print(grid.best_params_)
            print("O Modelo %s atingiu: %.2f%%  | %s" % (algoritmo, (grid.best_score_ * 100.0), alg_desc))
            
            scores.append((grid.best_score_, algoritmo, grid.best_params_, grid.best_estimator_ ))
    return scores




def gerar_submissao(clf, X, Y, X_test):
    clf = clf
    clf.fit(X, Y)

    X_test = X_test
    predictions = clf.predict(X_test)

    df_resultado = pd.DataFrame({'classe' : predictions})
    df_resultado.index = np.arange(1, len(df_resultado) +1)
    df_resultado.index.rename('id', inplace=True)
    df_resultado
    df_resultado.to_csv('sampleSubmission.csv', sep=',', encoding='utf-8')




list_alg = top_algoritmos(scores).keys()

best_datasets = []
best_datasets.append((X_iqr_fs_norm_mms, Y_iqr, "MinMaxScaler_FeatureSelection_e_IQR"))
best_datasets.append((X_iqr_fs_norm_ss,Y_iqr, "StandardScaler_FeatureSelection_e_IQR"))
best_datasets.append((X_iqr_fs, Y_iqr, "FeatureSelection_e_IQR"))
best_datasets.append((X_zscore_fs, Y_zscore, "FeatureSelection_e_Z-Score"))
best_datasets.append((X_zscore_fs_norm_mms, Y_zscore, "MinMaxScaler_FeatureSelection_e_Z-Score"))
best_datasets.append((X_zscore_fs_norm_ss, Y_zscore, "StandardScaler_FeatureSelection_e_Z-Score"))

scores_grid = grid_search(best_datasets, list_alg)    




#{'alpha': 0.1, 'solver': 'lbfgs', 'max_iter': 1000, 'random_state': 6, 'hidden_layer_sizes': 10}
#O Modelo MLP atingiu: 80.12%  | FeatureSelection_e_IQR

clf = MLPClassifier(alpha=0.1, solver='lbfgs', max_iter=1000, random_state=6, hidden_layer_sizes= 10)
cross = cross_val_score(clf, X_iqr_fs, Y_iqr, cv=10)
print("O Modelo de MLPClassifier atingiu: %.2f%% " % (cross.mean() * 100.0))




#{'penalty': 'l1', 'C': 1.0}
#O Modelo LR atingiu: 79.31%  | MinMaxScaler_FeatureSelection_e_IQ

clf = LogisticRegression(penalty='l1', C=1.0)
cross = cross_val_score(clf, X_iqr_fs_norm_mms, Y_iqr, cv=10)
print("O Modelo de LogisticRegression atingiu: %.2f%% " % (cross.mean() * 100.0))




# {'loss': 'deviance', 'subsample': 0.95, 'learning_rate': 0.2, 'min_samples_leaf': 0.1, 'n_estimators': 10, 
# 'min_samples_split': 0.2090909090909091, 'criterion': 'friedman_mse', 'max_features': 'log2', 'max_depth': 3}

GradientBoostingClassifier(loss='deviance', 
                           subsample=0.95, 
                           learning_rate=0.2, 
                           min_samples_leaf=0.1, 
                           n_estimators=10, 
                           min_samples_split=0.2090909090909091, 
                           criterion='friedman_mse', 
                           max_features='log2', 
                           max_depth=3)
cross = cross_val_score(clf, X_iqr_fs, Y_iqr, cv=10)
print("O Modelo de GradientBoostingClassifier atingiu: %.2f%% " % (cross.mean() * 100.0))




# preparando o dataset de treino para submissao

X_test = df_teste

best_columns = ['glicose', 'bmi', 'indice_historico', 'idade']
X_test_fs = X_test[best_columns]
X_test_fs_mms = normalizar_MinMaxScaler(X_test_fs)

#(clf, X, Y, X_test):
gerar_submissao(clf, X_iqr_fs, Y_iqr, X_test_fs_mms)




print(scores_grid)

