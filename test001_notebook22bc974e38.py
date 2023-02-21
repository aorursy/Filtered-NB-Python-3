#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




#Import the files you want to work with
#dfchat = pd.read_csv('../input/chat.csv')
#dfhn = pd.read_csv('../input/hero_names.csv')
dfmatch = pd.read_csv('../input/match.csv')
#dfobjectives = pd.read_csv('../input/objectives.csv')
#dfplt = pd.read_csv('../input/player_time.csv')
#dfpl = pd.read_csv('../input/players.csv')
#dftf = pd.read_csv('../input/teamfights.csv')
#dftfp = pd.read_csv('../input/teamfights_players.csv')
#dftl = pd.read_csv('../input/test_labels.csv')
#dftp = pd.read_csv('../input/test_player.csv')
dfpurl = pd.read_csv('../input/purchase_log.csv')
#dfabil = pd.read_csv('../input/ability_upgrades.csv')
#dfabilid = pd.read_csv('../input/ability_ids.csv')

print("ok")




dfpurl.head()




dfmatch.head()




dftf.head()




dftfp.head()




total = pd.concat([dfmatch,dfpurl], axis=0)
print("done")




print ("The new total dataset has {} data points with {} variables each." .format(*total.shape))




total.head()




total.shape




pd.isnull(total).sum() > 0




total.fillna(0)
#total = total.fillna(total.mean()) use this if you want to fill the NaN columns with mean of column not zero




total.loc[total["radiant_win"] == "False", "radiant_win"] = 0
total.loc[total["radiant_win"] == "True", "radiant_win"] = 1
total["radiant_win"] = total["radiant_win"].fillna("3")




print(total["radiant_win"].unique())




import colorsys
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
print("done")




fig,ax = plt.subplots(figsize=(8,5))
total['deaths'].value_counts(sort=False).plot(kind='bar',ax=ax,rot =90)
plt.title('deaths Distribution',fontsize=15)
plt.xlabel('deaths',fontsize=15)
plt.ylabel('negative_votes',fontsize=15)




duration = Counter(total['duration'].dropna().tolist()).most_common(10)
duration_name = [name[0] for name in duration]
duration_counts = [name[1] for name in duration]

fig,ax = plt.subplots(figsize=(8,5))
sns.barplot(x=duration_name,y=duration_counts,ax=ax)
plt.title('Top ten duration',fontsize=15)
plt.xlabel('duration',fontsize=15)
plt.ylabel('first_blood_time',fontsize=15)
ticks = plt.setp(ax.get_xticklabels(),fontsize=15,rotation=60)




total.corr()['buybacks']




total.corr()['gold_delta']




total.corr()['game_mode']




import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import ShuffleSplit
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
print("ok")




#this code crashed the kernel for timing out
#pd.scatter_matrix(total, alpha = 0.3,figsize = (20,20), diagonal = 'kde');




#this code crashed the kernal for exceeding memory
#total.plot.box()




from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis




h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)




datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

