#!/usr/bin/env python
# coding: utf-8



import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns




def visualize_classifier(classifier, X, y, title=''):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    # Define the mesh grid of X and Y values
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Run the classifier on the mesh grid
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Reshape the output array
    output = output.reshape(x_vals.shape)

    # Create a plot
    plt.figure(figsize=(12, 8))

    # Specify the title
    plt.title(title)

    # Choose a color scheme for the plot 
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c='yellow', s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # Specify the boundaries of the plot
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()


class_0 = dfData[ dfData['d'] == 0 ]
class_0 = class_0[ ['X', 'Y'] ].values




class_1 = dfData[ dfData['d'] == 1 ]
class_1 = class_1[ ['X', 'Y'] ].values




class_2 = dfData[ dfData['d'] == 2 ]
class_2 = class_2[ ['X', 'Y'] ].values




#Plote um gráfico scatter para visualização dos dados
plt.figure(figsize=(12, 8))
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', edgecolors='black', linewidth=1, marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', edgecolors='black', linewidth=1, marker='^')
plt.title('Dados de entrada')



params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}


visualize_classifier(classifier1, X_train, y_train, 'Dataset de treinamento')




visualize_classifier(classifier2, X_train, y_train, 'Trainingdataset')




visualize_classifier(classifier1, X_test, y_test, 'Classificador (Random Forest): Dataset de Validação')




visualize_classifier(classifier2, X_test, y_test, 'Classificador (Extremely Random Forest): Dataset de Validação')


# Evaluate classifier performance
class_names = ['Class-0', 'Class-1', 'Class-2']
print("\n" + "#"*40)
print("\nDesempenho do classificador (Random Forest) para o conjunto de treinamento\n")
# print(classification_report(y_train, classifier1.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

plt.show()




# Evaluate classifier performance
class_names = ['Class-0', 'Class-1', 'Class-2']
print("\n" + "#"*40)
print("\nDesempenho do classificador (Extremely Random Forest) para o conjunto de treinamento\n")
# print(classification_report(y_train, classifier2.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

plt.show()




print("#"*40)
print("\nDesempenho do classificador (Random Forest) para o conjunto de teste\n")
# print(classification_report(y_test, y_test_pred1, target_names=class_names))
print("#"*40 + "\n")

plt.show()




print("#"*40)
print("\nDesempenho do classificador (Extremely Random Forest) para o conjunto de teste\n")
# print(classification_report(y_test, y_test_pred2, target_names=class_names))
print("#"*40 + "\n")

plt.show()

