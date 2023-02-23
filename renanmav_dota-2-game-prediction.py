#!/usr/bin/env python
# coding: utf-8



#Loading the libraries
import numpy as np #Math library
import pandas as pd #Dataset library
import seaborn as sns #Graph library
import matplotlib.pyplot as plt #Help seaborn

#Importing data and renaming columns for consistency
df = pd.read_csv('../input/dota_games.txt', header=None)
df = df.rename(columns={0: 'ancient_1', 1: 'ancient_2', 2: 'ancient_3', 3: 'ancient_4', 4: 'ancient_5',
                        5: 'dire_1', 6: 'dire_2', 7: 'dire_3', 8: 'dire_4', 9: 'dire_5', 
                    10: 'team_win'})




from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report




print(df.info())




#Looking unique values
print(df.nunique())




#Knowing the data
print(df.head())




import plotly.offline as py #library that implement interactive visuals
py.init_notebook_mode(connected=True) #allow us to work with offline plotly
import plotly.graph_objs as go #like "plt" of matplotlib
import plotly.tools as tls #it will be useful soon

trace0 = go.Bar(
    x = df[df['team_win'] == 1]['team_win'].value_counts().index.values,
    y = df[df['team_win'] == 1]['team_win'].value_counts().values,
    name = 'Ancient team'
)

trace1 = go.Bar(
    x = df[df['team_win'] == 2]['team_win'].value_counts().index.values,
    y = df[df['team_win'] == 2]['team_win'].value_counts().values,
    name = 'Dire team'
)

data = [trace0, trace1]

layout = go.Layout(
    yaxis=dict(title='Wins'),
    xaxis=dict(title='Team'),
    title='Target variable distribution'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='grouped-bar')




y = df['team_win']
X = df.drop(['team_win'], axis=1)




print('The number of wins are equal to each team? {}'.format(
    round(len(df.loc[df.team_win == 1])/len(df.loc[df.team_win == 2]), 1) == 1
))
print('How much is this advantage ratio? {}%'.format(
    round(len(df.loc[df.team_win == 1])/len(df.loc[df.team_win == 2]) - 1, 3) * 100
))




winners_team_1 = df.loc[df.team_win == 1][['ancient_1', 'ancient_2', 'ancient_3', 'ancient_4', 'ancient_5']]
winners_team_1.rename(index=str, inplace=True, columns={'ancient_1': 'player_1', 
                                                        'ancient_2': 'player_2', 
                                                        'ancient_3': 'player_3', 
                                                        'ancient_4': 'player_4', 
                                                        'ancient_5': 'player_5'})

winners_team_2 = df.loc[df.team_win == 2][['dire_1', 'dire_2', 'dire_3', 'dire_4', 'dire_5']]
winners_team_2.rename(index=str, inplace=True, columns={'dire_1': 'player_1',
                                                        'dire_2': 'player_2',
                                                        'dire_3': 'player_3',
                                                        'dire_4': 'player_4',
                                                        'dire_5': 'player_5'})

winners = winners_team_1.append(winners_team_2)

hero_wins = winners.player_1.value_counts() +             winners.player_2.value_counts() +             winners.player_3.value_counts() +             winners.player_4.value_counts() +             winners.player_5.value_counts()

hero_wins = hero_wins.sort_values(ascending=False)

# TODO: get the wins for each hero




le = LabelEncoder()

for col in X.columns.values:
    le.fit(X[col].values)
    X[col] = le.transform(X[col])
    
print(X.info())




steps = [('scaler', StandardScaler()), ('logistic', SGDClassifier())]





alpha_space = np.logspace(-5, 8, 11)
param_grid = {'logistic__alpha': alpha_space}

cv = GridSearchCV(pipeline, param_grid, cv=5)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cv.fit(X_train, y_train)




y_pred = cv.predict(X_test)




print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
print("Tuned Model Score: {}".format(cv.best_score_))

