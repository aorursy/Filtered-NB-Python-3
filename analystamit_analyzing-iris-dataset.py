#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
pd.options.display.max_rows = 8
import warnings
warnings.filterwarnings("ignore")
#load dataset
iris = pd.read_csv('../input/Iris.csv', index_col = 0)
# Any results you write to the current directory are saved as output.




#import bokeh for visualization
from bokeh.io import output_notebook, show
from bokeh.charts import Histogram, Scatter, BoxPlot, Bar, HeatMap, bins
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
output_notebook()




print(np.shape(iris))
iris




## Histogram
iris_hp = Histogram(iris, values="PetalLengthCm", color="Species", 
                       legend="top_right", bins=12, width=800, height=400)

show(iris_hp)




#Scatter plot
iris_sp = Scatter(iris, x='PetalLengthCm', y='SepalLengthCm', color='Species', width=800, height=400)
show(iris_sp)




# Box plot
iris_bp = BoxPlot(iris, label='Species', values='PetalWidthCm', color='Species',
                     width=800, height=400, xlabel='', ylabel='petal width, cm',
                     title='Distributions of petal widths')
show(iris_bp)




iris_cds = ColumnDataSource(iris)

print(iris.Species.unique())

cmap = dict(zip(iris.Species.unique(), 'red green blue'.split()))
colors = [cmap[s] for s in iris.Species]




p = figure(width=800, height=400)
p.circle('PetalLengthCm', 'PetalWidthCm', color=colors, size=10, alpha=0.3, source=iris_cds)
p.xaxis.axis_label = 'petal_length'
p.yaxis.axis_label = 'petal_width'
show(p)




iris.columns




iris['Sepal_Area'] = iris['SepalLengthCm']*iris['SepalWidthCm']




iris['Petal_Area'] = iris['PetalLengthCm']*iris['PetalWidthCm']




#Importing more package for visualization against area
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "Sepal_Area", "Petal_Area")    .add_legend()




sns.boxplot(x="Species", y="Petal_Area", data=iris)




sns.boxplot(x="Species", y="Sepal_Area", data=iris)




iris.columns




def convert(a):
    if a == 'Iris-virginica':
        return 0
    elif a == 'Iris-versicolor':
        return 1
    else:
        return 2




X = iris[['PetalLengthCm','PetalWidthCm']]
iris['Species'] = iris['Species'].apply(convert)
y = iris['Species']




import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=0)

print('There are {} samples in the training set and {} samples in the test set'.format(
X_train.shape[0], X_test.shape[0]))
print()




from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print('After standardizing our features, the first 5 rows of our data now look like this:\n')
print(pd.DataFrame(X_train_std, columns=X_train.columns).head())




svc = svm.SVC(kernel = 'linear').fit(X_train_std, y_train)
print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svc.score(X_train_std, y_train)))

print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svc.score(X_test_std, y_test)))
rbf_svc = svm.SVC(kernel = 'poly').fit(X_train_std, y_train)
print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(rbf_svc.score(X_train_std, y_train)))

print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(rbf_svc.score(X_test_std, y_test)))
nu_svc = svm.NuSVC(kernel = 'linear').fit(X_train_std, y_train)
print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(nu_svc.score(X_train_std, y_train)))

print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(nu_svc.score(X_test_std, y_test)))
lin_svc = svm.LinearSVC().fit(X_train_std, y_train)
print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(lin_svc.score(X_train_std, y_train)))

print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(lin_svc.score(X_test_std, y_test)))




import warnings
from matplotlib.colors import ListedColormap

def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)




plot_decision_regions(X_test_std, y_test, svc)




y_test




y_test = y_test.map(a: lambda(a) = if a == 'Iris-virginica' then 0
                       elif a == 'Iris-versicolor' then 1
                       else 2)




convert = lambda a: lambda(a) = if a == 'Iris-virginica' then 0
                       elif a == 'Iris-versicolor' then 1
                       else 2




def convert(a):
    if a == 'Iris-virginica':
        return 0
    elif a == 'Iris-versicolor':
        return 1
    else:
        return 2




y_test = pd.series(map(convert, y_test))




y_test.to






