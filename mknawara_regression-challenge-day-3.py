#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error, r2_score
# GFX
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from subprocess import check_output
print(check_output(["ls", "../input/1000-cameras-dataset"]).decode("utf8"))




cameras = pd.read_csv("../input/1000-cameras-dataset/camera_dataset.csv")
cameras.head(5).transpose()
print("Shape: %s rows, %s columns" % (cameras.shape[0], cameras.shape[1]))

# nas = cameras.isnull().sum(axis=0)

df = pd.concat([cameras.isnull().sum(axis=0),                 cameras.applymap(lambda x: True if x==0 else False).sum(axis=0),                 cameras.dtypes], axis = 1)
df.columns = ["#NA", "#0s", "dtypes"]
print(df)




na_idxs = cameras["Macro focus range"].loc[cameras["Macro focus range"].isnull()].index.values[0]
print(cameras[(na_idxs-2):(na_idxs+2)])
cameras_orig = cameras
cameras.dropna(axis=0, how="any",inplace=True)




num_features = cameras.columns[1:]
fig, axs = plt.subplots(4,3,figsize=(15,13))
axs = axs.flatten()

i = 0
for feature in num_features:
    sns.distplot(cameras[feature], ax = axs[i],                  color=(sns.color_palette()[i % len(sns.color_palette())]));
    i += 1




corr = cameras[num_features].corr()
cg = sns.clustermap(corr, cmap="YlGnBu");
plt.show();




corr = cameras[num_features].corr("spearman")
cg = sns.clustermap(corr, cmap="RdYlGn");
plt.show();
corr_feats = ['Price', 'Weight (inc. batteries)', 'Dimensions', 'Zoom tele (T)', 'Zoom wide (W)', 'Macro focus range']




ax = sns.pairplot(cameras[corr_feats], palette = "Spectral", hue="Price");
ax._legend.remove()
plt.show();




def nol(df, feature, m=2):
    if m==0:
        return df
    x = df[feature]
    mask = abs(x - np.mean(x)) < m*np.std(x)
    return df.loc[mask]

reg_feats = list(corr_feats[i] for i in [1,2,4,5])
m_=1.5 # No. of std deviations - cutoff
cameras_nol = nol(cameras, 'Price', m=m_)

fig, axs = plt.subplots(2,2,figsize=(15,10))
axs = axs.flatten()
i=0
for feature in reg_feats:
    df = cameras_nol
    ax = sns.regplot(x=feature, y='Price', data=df, ax=axs[i])
    left, right = min(df[feature]) - 0.3*df[feature].std(),        max(df[feature] + 0.3*df[feature].std())
    ax.set_xlim(left, right)
    i+=1




def nozero(df, feature):
    x = df[feature]
    mask = x != 0
    return df.loc[mask]

reg_feats = list(corr_feats[i] for i in [1,2,4,5])
m_=1.5 # No. of std deviations - cutoff

fig, axs = plt.subplots(2,2,figsize=(15,10))
axs = axs.flatten()
i=0
cameras_nol = nol(cameras, 'Price', m=m_)
for feature in reg_feats:
    df = nozero(cameras_nol, feature)
    ax = sns.regplot(x=feature, y='Price', data=df, ax=axs[i], marker='.')
    left, right = min(df[feature]) - 0.3*df[feature].std(),        max(df[feature] + 0.3*df[feature].std())
    ax.set_xlim(left, right)
    i+=1




from sklearn.cluster import KMeans
prices = np.array(cameras['Price'])
colors = ["blue", "green", "yellow", "orange"]
fig, ax = plt.subplots(figsize=(15,6))
# optional - logscale (really saving for later projs)
# ax = plt.subplot(111)
# ax.set_yscale("log")
for k in range(2,5):
    km = KMeans(n_clusters=k).fit(prices.reshape(-1,1))
    cluster = km.labels_
    for i in range(0,k):
        y=prices[cluster==i]
        plt.scatter(x=[k]*len(y)+np.random.normal(0,0.01,(len(y))),                     y=y, c=sns.color_palette()[i], marker='.', alpha=0.8);
ax.set(xlabel="# clusters", xticks=[2,3,4], xticklabels=[2,3,4],       ylabel="Price", title="KMeans price clusters")
plt.show()
print("Look at k=4 for filtering:")
km = KMeans(n_clusters=4).fit(prices.reshape(-1,1))
cameras['Cluster'] = km.labels_
print("Group %s: %s\n"*4 % sum(tuple((i+1, (km.labels_ == i).sum()) for i in range(4)),()))




fig, axs = plt.subplots(2,2,figsize=(15,10))
axs = axs.flatten()
i=0
k=0
for feature in reg_feats:
    df = nozero(cameras, feature)
    df = df.loc[df['Cluster']==k]
    ax = sns.regplot(x=feature, y='Price', data=df, ax=axs[i], marker='.')
    left, right = min(df[feature]) - 0.3*df[feature].std(),        max(df[feature] + 0.3*df[feature].std())
    ax.set_xlim(left, right)
    i+=1
print("cluster filtering looks good, but at this zoom the linear relationships look fairly flat...")




other_reg_feats = ['Max resolution', 'Effective pixels', 'Storage included', 'Normal focus range']
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs = axs.flatten()
i=0
k=0
for feature in other_reg_feats:
    df = nozero(cameras, feature)
    df = df.loc[df['Cluster']==k]
    ax = sns.regplot(x=feature, y='Price', data=df, ax=axs[i], marker='.')
    left, right = min(df[feature]) - 0.3*df[feature].std(),        max(df[feature] + 0.3*df[feature].std())
    ax.set_xlim(left, right)
    i+=1




feature = 'Max resolution'
fig, axs = plt.subplots(1,2,figsize=(15,5))
axs = axs.flatten()
i = 0
for k in [0,1]:
    df = nozero(cameras, feature)
    df = df.loc[df['Cluster']==k]
    ax = sns.regplot(x=feature, y='Price', data=df, ax=axs[i], marker='.')
    left, right = min(df[feature]) - 0.3*df[feature].std(),        max(df[feature] + 0.3*df[feature].std())
    ax.set_xlim(left, right)
    i += 1




import statsmodels.api as sm
import statsmodels.formula.api as smf

df = nozero(cameras, feature)
df = df.loc[df['Cluster'] == 0]
X = df[feature]
X = sm.tools.add_constant(X, prepend=True)
Y = df['Price']
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())




n = len(results.resid)
ax = plt.figure(figsize = (12,6))
plt.scatter(x=np.linspace(0,n,n), y=results.resid, marker='.');
l = plt.axhline(0, color="green", ls="--");




from scipy.stats import probplot
print("QQ plot:")
ax = plt.figure(figsize = (12,6))
probplot(results.resid, plot=plt)
plt.show()

