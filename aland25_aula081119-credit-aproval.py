#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




#dados = pd.read_csv('/kaggle/input/creditscreening/credit-screening.data')
nomes = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18"] # renomeia os titulos das colunas
df = pd.read_csv('/kaggle/input/creditscreening/credit-screening.data', sep=",", names=nomes, na_values="?") # na_values="?" troca valor ? p/ Nan 
df.head(690) # mostra a tabela com as 690 linhas e as 16 colunas + duas criadas 18




df.describe() # mostra um resumo da tabela 




df.isnull().sum() # mostra que na coluna A1 tem 12NaN  colunas A2=12Nan A4=6NaN sucessivamente ... A14=13NaN  




cols = ['A2']
 for i in cols:
    df[i].fillna(value=df[i].moda())
    df.isnull().sum()




# aIterar sobre cada coluna de df
for col in df.columns:
    # Verifique se a coluna é do tipo de objeto
    if df[col].dtypes == 'object':
       # Imputar com o valor mais frequente
        df = df.fillna(df[col].value_counts().index[0])
# Conte o número de NaNs no conjunto de dados e imprima as contagens para verificar
print(df.isnull().sum())




cols = ['A2','A14']
 for i in cols:
    df[i].fillna(value=df[i].mean, implace=True)
    df.isnull().sum()









cols = ['A1','A4','A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']
for i in cols:
    df[i].astype('category')




cols = ['A16']
for i in cols:
    df['A16'].cat.codes














































