#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import itertools
import seaborn as sns
import statsmodels.api as sm
from patsy.contrasts import Diff
sns.set_style('white')

df = pd.read_csv("../input/catanstats.csv")
df.me.fillna(0, inplace=True)




# Here I make a bunch of features that I think might be predictive. These include:

# 1. The expected number of cards (ec) on any roll
# 2. The ability to build items without needing to trade
# 3. Having ports that you can use




# turn the numbers that a player is on into a probability of resources per roll
probs = defaultdict(int)
for d1, d2 in itertools.combinations_with_replacement(range(1, 7), 2):
    s = d1 + d2
    probs[s] += 1 if d1 == d2 else 2
probs = {k:v/sum(probs.values()) for k, v in probs.items()}

resources = "LCSWO"
settlement_res = set("LCSW")
road_res = set("WC")
dcard_res = set("SOW")

def get_row_ec(row, cards=resources):
    ec = 0
    nums = row[15:27].tolist()[0::2]
    vals = row[15:27].tolist()[1::2]
    for n, v in zip(nums, vals):
        if v in cards:
            ec += probs[n]
    return ec

def two_port(row):
    # two ports are only considered useful if you sit on a resource that it trades
    vals = row[15:27].tolist()[1::2]
    s = sum(1 for v in vals if v[0] == '2' and v[1] in vals)
    return s

def three_port(row):
    vals = row[15:27].tolist()[1::2]
    s = sum(1 for v in vals if v[0] == '3')
    return s

def city(row):
    # fast city-building if there are more than 1 ore and a wheat
    vals = row[15:27].tolist()[1::2]
    if vals.count('O') >= 2 and 'W' in vals:
        return 1
    return 0

def can_build(row, req):
    # see if the adjacent tiles contain the input set
    vals = row[15:27].tolist()[1::2]
    if req.issubset(vals):
        return 1
    return 0




# get the features
df['init_ec'] = df.apply(get_row_ec, axis=1)
df['two_port'] = df.apply(two_port, axis=1)
df['three_port'] = df.apply(three_port, axis=1)
df['city'] = df.apply(city, axis=1)
df['settlement'] = df.apply(can_build, axis=1, args=(settlement_res,))
df['road'] = df.apply(can_build, axis=1, args=(settlement_res,))
df['dcard'] = df.apply(can_build, axis=1, args=(dcard_res,))




## Expected Cards vs. Points##
The initial expected number of cards per turn correlates with points significantly with a value of 0.29




g = sns.jointplot(df.init_ec, df.points, alpha=0.5);




model_eqn = "I(points-2) ~ 1 + (init_ec + C(dcard) + C(me))"
model = sm.OLS.from_formula(model_eqn, df).fit()
print(model.summary())




mn, mx = 1, 13
plt.scatter(model.model.endog+2, model.fittedvalues+2);
plt.xlim(mn, mx); plt.ylim(mn, mx); plt.plot([mn, mx], [mn, mx], '-k', lw=1);
plt.ylabel("Fit Value"); plt.xlabel("Actual Value"); plt.title("Actual by Fit");




# break EV up by card type
inputs = []
for card in "LCSWO":
    inputs.append('ec_' + card)
    df['ec_' + card] = df.apply(get_row_ec, axis=1, args=(card,))




model_eqn = "I(points - 2) ~ -1 + (" + " + ".join(inputs) + ")" #**2
model = sm.OLS.from_formula(model_eqn, df).fit()
print(model.summary())




mn, mx = 1, 13
plt.scatter(model.model.endog+2, model.fittedvalues+2);
plt.xlim(mn, mx); plt.ylim(mn, mx); plt.plot([mn, mx], [mn, mx], '-k', lw=1);
plt.ylabel("Fit Value"); plt.xlabel("Actual Value"); plt.title("Actual by Fit");




df['win'] = 0
wincol = df.columns.tolist().index('win')

for gnum in df.gameNum.unique():
    rows = df[df.gameNum == gnum]
    win_idx = rows.points.argmax()
    df.iloc[win_idx, wincol] = 1




# First, predict the winner based only on the player position and whether or not our kind data
# gatherer was the player
model_eqn = "win ~ 1 + C(me) + C(player)"
model = sm.Logit.from_formula(model_eqn, df).fit()
print(model.summary())




t = model.pred_table()
print(t)
print("Accuracy:",np.diag(t).sum()/t.sum())




def model_pred(model, data):
    right = 0
    tried = 0
    for gnum in df.gameNum.unique():
        print(gnum)
        rows = df[df.gameNum == gnum]
        win_idx = rows.points.argmax()
        max_idx = -1
        max_pts = 0
        pred_max_idx = -1
        pred_max_p = -1
        for idx, r in rows.iterrows():
            p = model.predict(r)
            if p > pred_max_p:
                pred_max_p = p
                pred_max_idx = idx
            if r.points > max_pts:
                max_pts = r.points
                max_idx = idx
        tried += 1
        right += 1 if max_idx == pred_max_idx else 0
    return right/tried




model_pred(model, df)




# make new data, where a single row is a full game
game_rows = []
for gnum in df.gameNum.unique():
    gamerow = []
    rows = df[df.gameNum == gnum]
    for idx, r in rows.iterrows():
        # Add whatever you think is predictive for each player here
        gamerow.extend([r.init_ec, r.me])
        if r.win:
            winner = idx % 4
    gamerow += [winner]
    game_rows.append(gamerow)
    
gamedf = pd.DataFrame(game_rows)
# I ignored column names and just renamed the last one to winner
gamedf.rename(columns={gamedf.shape[1]-1:'WINNER'}, inplace=True)
# Rename all the default numbered columns to 'V#'
gamedf.rename(columns={x: "V{}".format(x) for x in gamedf.columns if x != 'WINNER'}, inplace=True)




inputs = [x for x in gamedf.columns if x != "WINNER"]
# NOTE: I cheat here a bit and don't set the `me` variable as categorical.
model_eqn = "WINNER ~ -1 + " + " + ".join(inputs)
model = sm.MNLogit.from_formula(model_eqn, gamedf).fit()
print(model.summary())




t = model.pred_table()
print(t)
print("Accuracy:",np.diag(t).sum()/t.sum())




from sklearn import linear_model
from sklearn.cross_validation import KFold, cross_val_score

X = gamedf[[x for x in gamedf.columns if x != 'WINNER']].as_matrix()
y = gamedf.WINNER

logreg = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')

# holdout one at a time.
kf = KFold(len(gamedf), n_folds=50, shuffle=False)
print("Average CV accuracy:", cross_val_score(logreg, X, y, cv=kf).mean())











