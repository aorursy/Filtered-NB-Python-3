#!/usr/bin/env python
# coding: utf-8



# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#plotly graphing
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import colorlover as cl

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.core.display import display, HTML, Javascript
from string import Template
import json
import IPython.display


import warnings
warnings.filterwarnings('ignore')




get_ipython().run_cell_magic('javascript', '', 'require.config({\n    paths: {\n        d3: "https://d3js.org/d3.v4.min"\n     }\n});\n\nrequire(["d3"], function(d3) {\n    window.d3 = d3;\n});')




train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
df = train_df # as we alter train_df later
combine = [train_df, test_df]




print(train_df.columns.values)




# preview the data
train_df.head()




train_df.tail()




train_df.info()
print('_'*40)
test_df.info()




train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`




train_df.describe(include=['O'])




train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)




train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)




train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)




train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)




g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)




def go_hist_trace(data, name, colour='rgba(171, 50, 96, 0.6)'):
    trace = go.Histogram(x=data,
                    opacity=0.75,
                    name = name,
                    xbins=dict(
                        start=0,
                        end=80,
                        size=4
                    ), 
                    autobinx = False,
                    marker=dict(color= colour))
    return trace




# import graph objects as "go"
age_survived = df[df.Survived == 1]['Age']
age_not_survived = df[df.Survived == 0]['Age']
#age_male = train_df[train_df.Sex.str.contains("male")]['Age']
  
trace1 = go_hist_trace(data=age_survived,name='Survived',colour='rgba(171, 50, 96, 0.6)')
trace2 = go_hist_trace(data=age_not_survived,name='Not survived',colour='rgba(12, 50, 196, 0.6)')

data = [trace1, trace2]
layout = go.Layout(barmode='overlay',
                   title='Survived vs not survived by age',
                   xaxis=dict(title='Survival ratio'),
                   yaxis=dict( title='Count'),
                  )
fig = dict(data = data, layout=layout)
iplot(fig)
#fig = go.create_distplot(hist_data, group_labels)
#py.iplot(fig, filename='Basic Distplot')

#from dataframe
#py.iplot(ff.create_distplot([train_df['Age']], 'Age', bin_size=0.25),
#                            filename='distplot with pandas')




# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();




df_age_plcass_not_survived = df.loc[df['Survived'] == 0,['Age','Pclass']]
df_age_plcass_survived = df.loc[df['Survived'] == 1,['Age','Pclass']]

age_not_survived_p1 = df_age_plcass_not_survived[df_age_plcass_not_survived.Pclass == 1]['Age']
age_not_survived_p2 = df_age_plcass_not_survived[df_age_plcass_not_survived.Pclass == 2]['Age']
age_not_survived_p3 = df_age_plcass_not_survived[df_age_plcass_not_survived.Pclass == 3]['Age']

age_survived_p1 = df_age_plcass_survived[df_age_plcass_survived.Pclass == 1]['Age']
age_survived_p2 = df_age_plcass_survived[df_age_plcass_survived.Pclass == 2]['Age']
age_survived_p3 = df_age_plcass_survived[df_age_plcass_survived.Pclass == 3]['Age']

# colours
bupu = cl.scales['9']['seq']['BuPu']


trace1 = go_hist_trace(data=age_not_survived_p1,name='Pclass 1 Not Survived',colour=bupu[2])
trace2 = go_hist_trace(data=age_not_survived_p2,name='Pclass 2 Not Survived',colour=bupu[3])
trace3 = go_hist_trace(data=age_not_survived_p3,name='Pclass 3 Not Survived',colour=bupu[4])
trace4 = go_hist_trace(data=age_survived_p1,name='Pclass 1 Survived',colour=bupu[6])
trace5 = go_hist_trace(data=age_survived_p2,name='Pclass 2 Survived',colour=bupu[7])
trace6 = go_hist_trace(data=age_survived_p3,name='Pclass 3 Survived',colour=bupu[8])

data = [trace1, trace2, trace3, trace4, trace5, trace6]
layout = go.Layout(barmode='overlay',
                   title='Survived vs not survived by age/Pclass',
                   xaxis=dict(title='Survival ratio'),
                   yaxis=dict( title='Count'),
                  )
fig = dict(data = data, layout=layout)
iplot(fig)




# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()




# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()




print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape




for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])




for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()




title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()




train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape




for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()




# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()




guess_ages = np.zeros((2,3))
guess_ages




for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()




train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)




for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()




train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()




for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)




for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()




train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()




for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)




freq_port = train_df.Embarked.dropna().mode()[0]
freq_port




for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)




for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()




test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()




train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)




for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)




test_df.head(10)




X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape




# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log




coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)




# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc




knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn




# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian




# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron




# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc




# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd




# Decision Tree

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree




# rules defined in the tree object clf
def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        #count_labels = zip(clf.tree_.value[node_index, 0], labels)
        #node['name'] = ', '.join(('{} of {}'.format(int(count), label)
        #                          for count, label in count_labels))
        node['type']='leaf'
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        node['error'] = np.float64(clf.tree_.impurity[node_index]).item()
        node['samples'] = clf.tree_.n_node_samples[node_index]
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['type']='split'
        node['label'] = '{} > {}'.format(feature, threshold)
        node['error'] = np.float64(clf.tree_.impurity[node_index]).item()
        node['samples'] = clf.tree_.n_node_samples[node_index]
        node['value'] = clf.tree_.value[node_index, 0].tolist()
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
        
    return node

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)




cols = X_train.columns
d = rules(decision_tree, cols, None)
with open('output.json', 'w') as outfile:  
    json.dump(d, outfile,cls=MyEncoder)

j = json.dumps(d, cls=MyEncoder)




html_string = """
<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>
    <script type="text/javascript" src="https://d3js.org/d3.v3.min.js"></script>
    <style type="text/css">
body {
  font-family: "Helvetica Neue", Helvetica;
}
.hint {
  font-size: 12px;
  color: #999;
}
.node rect {
  cursor: pointer;
  fill: #fff;
  stroke-width: 1.5px;
}
.node text {
  font-size: 11px;
}
path.link {
  fill: none;
  stroke: #ccc;
}
    </style>
  </head>
  <body>
    <div id="body">
      <div id="footer">
        Decision Tree viewer
        <div class="hint">click to expand or collapse</div>
        <div id="menu">
          <select id="datasets"></select>
        </div>

      </div>
    </div>    
"""




js_string="""
 var m = [20, 120, 20, 120],
    w = 1280 - m[1] - m[3],
    h = 800 - m[0] - m[2],
    i = 0,
    rect_width = 80,
    rect_height = 20,
    max_link_width = 20,
    min_link_width = 1.5,
    char_to_pxl = 6,
    root;
// Add datasets dropdown
d3.select("#datasets")
    .on("change", function() {
      if (this.value !== '-') {
        d3.json(this.value + ".json", load_dataset);
      }
    })
  .selectAll("option")
    .data([
      "-",
      "output"
    ])
  .enter().append("option")
    .attr("value", String)
    .text(String);
var tree = d3.layout.tree()
    .size([h, w]);
var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.x, d.y]; });
var vis = d3.select("#body").append("svg:svg")
    .attr("width", w + m[1] + m[3])
    .attr("height", h + m[0] + m[2] + 1000)
  .append("svg:g")
    .attr("transform", "translate(" + m[3] + "," + m[0] + ")");
// global scale for link width
var link_stoke_scale = d3.scale.linear();
var color_map = d3.scale.category10();
// stroke style of link - either color or function
var stroke_callback = "#ccc";
function load_dataset(json) {
  root = json;
  root.x0 = 0;
  root.y0 = 0;
  var n_samples = root.samples;
  var n_labels = root.value.length;
  if (n_labels >= 2) {
    stroke_callback = mix_colors;
  } else if (n_labels === 1) {
    stroke_callback = mean_interpolation(root);
  }
  link_stoke_scale = d3.scale.linear()
                             .domain([0, n_samples])
                             .range([min_link_width, max_link_width]);
  function toggleAll(d) {
    if (d && d.children) {
      d.children.forEach(toggleAll);
      toggle(d);
    }
  }
  // Initialize the display to show a few nodes.
  root.children.forEach(toggleAll);
  update(root);
}
function update(source) {
  var duration = d3.event && d3.event.altKey ? 5000 : 500;
  // Compute the new tree layout.
  var nodes = tree.nodes(root).reverse();
  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * 180; });
  // Update the nodes…
  var node = vis.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });
  // Enter any new nodes at the parent's previous position.
  var nodeEnter = node.enter().append("svg:g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + source.x0 + "," + source.y0 + ")"; })
      .on("click", function(d) { toggle(d); update(d); });
  nodeEnter.append("svg:rect")
      .attr("x", function(d) {
        var label = node_label(d);
        var text_len = label.length * char_to_pxl;
        var width = d3.max([rect_width, text_len])
        return -width / 2;
      })
      .attr("width", 1e-6)
      .attr("height", 1e-6)
      .attr("rx", function(d) { return d.type === "split" ? 2 : 0;})
      .attr("ry", function(d) { return d.type === "split" ? 2 : 0;})
      .style("stroke", function(d) { return d.type === "split" ? "steelblue" : "olivedrab";})
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });
  nodeEnter.append("svg:text")
      .attr("dy", "12px")
      .attr("text-anchor", "middle")
      .text(node_label)
      .style("fill-opacity", 1e-6);
  // Transition nodes to their new position.
  var nodeUpdate = node.transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
  nodeUpdate.select("rect")
      .attr("width", function(d) {
        var label = node_label(d);
        var text_len = label.length * char_to_pxl;
        var width = d3.max([rect_width, text_len])
        return width;
      })
      .attr("height", rect_height)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });
  nodeUpdate.select("text")
      .style("fill-opacity", 1);
  // Transition exiting nodes to the parent's new position.
  var nodeExit = node.exit().transition()
      .duration(duration)
      .attr("transform", function(d) { return "translate(" + source.x + "," + source.y + ")"; })
      .remove();
  nodeExit.select("rect")
      .attr("width", 1e-6)
      .attr("height", 1e-6);
  nodeExit.select("text")
      .style("fill-opacity", 1e-6);
  // Update the links
  var link = vis.selectAll("path.link")
      .data(tree.links(nodes), function(d) { return d.target.id; });
  // Enter any new links at the parent's previous position.
  link.enter().insert("svg:path", "g")
      .attr("class", "link")
      .attr("d", function(d) {
        var o = {x: source.x0, y: source.y0};
        return diagonal({source: o, target: o});
      })
      .transition()
      .duration(duration)
      .attr("d", diagonal)
      .style("stroke-width", function(d) {return link_stoke_scale(d.target.samples);})
      .style("stroke", stroke_callback);
  // Transition links to their new position.
  link.transition()
      .duration(duration)
      .attr("d", diagonal)
      .style("stroke-width", function(d) {return link_stoke_scale(d.target.samples);})
      .style("stroke", stroke_callback);
  // Transition exiting nodes to the parent's new position.
  link.exit().transition()
      .duration(duration)
      .attr("d", function(d) {
        var o = {x: source.x, y: source.y};
        return diagonal({source: o, target: o});
      })
      .remove();
  // Stash the old positions for transition.
  nodes.forEach(function(d) {
    d.x0 = d.x;
    d.y0 = d.y;
  });
}
// Toggle children.
function toggle(d) {
  if (d.children) {
    d._children = d.children;
    d.children = null;
  } else {
    d.children = d._children;
    d._children = null;
  }
}
// Node labels
function node_label(d) {
  if (d.type === "leaf") {
    // leaf
    var formatter = d3.format(".2f");
    var vals = [];
    d.value.forEach(function(v) {
        vals.push(formatter(v));
    });
    return "[" + vals.join(", ") + "]";
  } else {
    // split node
    return d.label;
  }
}
/**
 * Mixes colors according to the relative frequency of classes.
 */
function mix_colors(d) {
  var value = d.target.value;
  var sum = d3.sum(value);
  var col = d3.rgb(0, 0, 0);
  value.forEach(function(val, i) {
    var label_color = d3.rgb(color_map(i));
    var mix_coef = val / sum;
    col.r += mix_coef * label_color.r;
    col.g += mix_coef * label_color.g;
    col.b += mix_coef * label_color.b;
  });
  return col;
}
/**
 * A linear interpolator for value[0].
 *
 * Useful for link coloring in regression trees.
 */
function mean_interpolation(root) {
  var max = 1e-9,
      min = 1e9;
  function recurse(node) {
    if (node.value[0] > max) {
      max = node.value[0];
    }
    if (node.value[0] < min) {
      min = node.value[0];
    }
    if (node.children) {
      node.children.forEach(recurse);
    }
  }
  recurse(root);
  var scale = d3.scale.linear().domain([min, max])
                               .range(["#2166AC","#B2182B"]);
  function interpolator(d) {
    return scale(d.target.value[0]);
  }
  return interpolator;
}
 """




h = display(HTML(html_string))
j = IPython.display.Javascript(js_string)
IPython.display.display_javascript(j)




# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest




models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)




submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)

