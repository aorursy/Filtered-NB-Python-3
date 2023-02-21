#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('pylab', 'inline')

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from __future__ import division




df = pd.read_csv("../input/free_throws.csv")
df.head(2)




games = df.drop_duplicates("game_id")           .groupby(["season", "playoffs"]).size()           .unstack()
games.head(3)




fig, ax = plt.subplots(1,2, figsize=(15,5))
plt.suptitle("Number of Games per Season", y=1.03, fontsize=20)

games.regular.plot(marker="o", rot=90, title="Regular Season", color="#41ae76", ax=ax[0])
games.playoffs.plot(marker="o", rot=90, title="Playoffs", ax=ax[1])




ft_total = df.groupby(["season", "playoffs"]).size()              .unstack()
ft_total.head(3)




ft_per_game = ft_total / games
ft_per_game.head(2)




ft_per_game.plot(marker="o", rot=90, figsize=(12,5))
plt.title("Average Number of Free Throws per Game", fontsize=20)

plt.arrow(5.3, 51, -0.5, -1.2, width=0.01, color="k", head_starts_at_zero=False)
plt.text(4.8, 51.2, "Change of Rules")




periods = df.groupby(["game_id", "playoffs", "period"]).size()             .unstack(["playoffs", "period"])             .describe()[:2]             .stack().unstack(0)             .swaplevel(0, 1, axis=1).sortlevel(axis=1)
            
periods




periods["mean"][:5].plot(marker="o", xticks=(1,2,3,4,5), xlim=(0.8, 5.2), figsize=(8,5))
plt.title("Average Number of Free Throws", fontsize=20)




periods["minutes"] = [12,12,12,12,5,5,5,5]
periods["playoffs"] = periods["mean"].playoffs / periods.minutes
periods["regular"] = periods["mean"].regular / periods.minutes
periods




per_minute = periods[["playoffs", "regular"]][:5]
per_minute.columns = per_minute.columns.droplevel(1)

per_minute.plot(marker="o", xticks=(1,2,3,4,5), xlim=(0.8, 5.2), figsize=(8,5))
plt.title("Average Number of Free Throws per Minute", fontsize=20)




# excluding free throws that were made during overtime
df_seconds_left = df[df.period <= 4]




def determining_seconds_left(row):
    
    minutes_left_period = int(row.time.split(":")[0])
    seconds_left_period = int(row.time.split(":")[1])
    
    remaining_periods = 4 - row.period
    remaining_seconds = remaining_periods * 12 * 60
    
    seconds_left_total = minutes_left_period * 60 + seconds_left_period + remaining_seconds
    
    return seconds_left_total




df_seconds_left["seconds_left"] = df_seconds_left.apply(determining_seconds_left, axis=1)




df_seconds_left.seconds_left.hist(bins=48, figsize=(15,4))

plt.xlabel("Seconds left")
plt.title("Number of Free Throws", fontsize=20)
plt.xticks(range(0, 3240, 360))
plt.xlim(0, 2880)

plt.vlines([12*60, 24*60, 36*60], 0, 40000, linestyle="--")
plt.text(250, 37000, "4th Quarter")
plt.text(970, 37000, "3rd Quarter")
plt.text(1690, 37000, "2nd Quarter")
plt.text(2410, 37000, "1st Quarter")




shooting = df.groupby(["player"])["shot_made"].agg(["size", "mean"])
shooting = shooting.rename(columns={"size": "ft_count", "mean": "percentage"})

# to make sure the shooting percentages are valid, the players should have at least taken 100 shots
shooting = shooting[shooting.ft_count>=100]

shooting.head(3)




shooting.percentage.hist(bins=50, figsize=(8,5))

plt.title("Distribution of Shooting Percentages", fontsize=20)
plt.xlabel("Shooting Percentage")

plt.vlines(x=shooting.percentage.median(), ymin=0, ymax=45, color="red", linestyle="--")
plt.text(x=0.72, y=-1.3, s="median", color="red")




shooting.sort_values(by="percentage", ascending=False)[:10]




shooting.sort_values(by="percentage")[:10]




ft_per_game = df.groupby(["player", "game_id"]).size()                 .unstack("player")                 .mean().sort_values(ascending=False)
        
# adding shooting percentages from the shooting dataframe
ft_per_game = pd.DataFrame({"ft_per_game": ft_per_game})
ft_per_game["percentage"] = shooting.percentage

# dropping those players that had less than 100 shots in the shooting dataframe
ft_per_game = ft_per_game.dropna()




ft_per_game.ft_per_game.hist(bins=50, figsize=(8,5))

plt.title("Distribution of Free Throws per Game per Player", fontsize=20)
plt.xlabel("Number of Free Throws")
plt.vlines(x=ft_per_game.ft_per_game.median(), ymin=0, ymax=70, color="red", linestyle="--")
plt.text(x=3.08, y=-2, s="median", color="red")




ft_per_game.head(10)




shooting_per_season = df.groupby(["player", "season"])["shot_made"].agg(["mean", "size"])

# player has at least 100 shots per season
shooting_per_season = shooting_per_season[shooting_per_season["size"]>=100]

# dropping level "size"
shooting_per_season = shooting_per_season.drop("size", axis=1).unstack("player")

# removing the hierarchical index "mean"
shooting_per_season.columns = shooting_per_season.columns.droplevel()

# there are at least 5 seasons of data
shooting_per_season = shooting_per_season.dropna(axis=1)

shooting_std = shooting_per_season.std()

# adding the overall shooting percentage as reference
shooting_std = pd.DataFrame({"std": shooting_std})
shooting_std["shooting_percentage"] = shooting.percentage

shooting_std.sort_values(by="std").head(10)




shooting_std.sort_values(by="std", ascending=False).head(10)




shooting_std.plot(kind="scatter", x="shooting_percentage", y="std", figsize=(8,5))
plt.title("Consistency of Shooting in relation to Shooting Percentage", fontsize=15)




most_inconsistent = shooting_std.sort_values(by="std", ascending=False).head(3).index
most_consistent = shooting_std.sort_values(by="std").head(3).index

fig, ax = plt.subplots(1,2, figsize=(20,5), sharey=True)

ax1 = shooting_per_season[most_consistent].plot(marker="o", rot=90, ax=ax[0], title="Top 3: Most Consistent")
ax2 = shooting_per_season[most_inconsistent].plot(marker="o", rot=90, ax=ax[1], title="Top 3: Most Inconsistent")

plt.setp(ax2.get_yticklabels(), visible=True)
plt.suptitle("Shooting Percentages over all 10 Seasons", y=1.03, fontsize=20)




def seconds_left(value):
    time = value.split(":")
    seconds = int(time[0]) * 60 + int(time[1])
    return seconds




df["seconds"] = df.time.apply(seconds_left)




df["previous_score"] = df.score.shift()




def who_shot(row):
    if ("makes free throw 2 of 2" in row.play) or ("makes free throw 3 of 3") in row.play:
        if row.score.split(" - ")[0] != row.previous_score.split(" - ")[0]:
            return row.game.split(" - ")[0]
        else:
            return row.game.split(" - ")[1]
    else:
        return np.nan




df["team"] = df.apply(who_shot, axis=1)




df.head(2)




players = {}

for index, row in df[df.team.notnull()].iterrows():
    try:
        players[row.player][row.game_id] = row.team
    except KeyError:
        players[row.player] = {row.game_id: row. team}




def who_shot(row):
    try:
        return players[row.player][row.game_id]
    except KeyError:
        return np.nan




df["team"] = df.apply(who_shot, axis=1)




def find_score_difference(row):
    
    teams = row.game.split(" - ")
    scores = row.score.split(" - ")
    
    if row.team == teams[0]:
        own_score = int(scores[0])
        opponent_score = int(scores[1])
        
        if row.shot_made == 1:
            own_score = own_score - 1

    elif row.team == teams[1]:
        own_score = int(scores[1])
        opponent_score = int(scores[0])
        
        if row.shot_made == 1:
            own_score = own_score - 1
    else:
        return np.nan
    
    return own_score - opponent_score




df["score_difference"] = df.apply(find_score_difference, axis=1)




df.score_difference.hist(bins=50, figsize=(8,5))

plt.title("Distribution of Score Differences", fontsize=20)
plt.xlabel("Score Difference")




print("Number of free throws for which the team is determined: ", len(df[df.team.notnull()]))
print("Percentage of free throws for which the team is determined: ", len(df[df.team.notnull()]) / len(df), "%")




high_pressure = df[(df.period>=4) & (df.seconds<=120) & (df.score_difference<=5) & (df.score_difference>=-5)]

high_pressure = high_pressure.groupby("player")["shot_made"].agg(["mean", "size"])
high_pressure = high_pressure.rename(columns={"mean": "percentage_pressure", "size": "count_pressure"})

high_pressure = high_pressure[high_pressure["count_pressure"] >= 100]




pressure = pd.merge(high_pressure, shooting, left_index=True, right_index=True)
pressure.head(3)




worst_percentage = pressure.percentage.min()

pressure.plot(kind="scatter", x="percentage", y="percentage_pressure", figsize=(8,5))
plt.plot([worst_percentage, 1], [worst_percentage, 1], color="red")

plt.title("Shooting Performance under Pressure", fontsize=20)
plt.xlabel("Shooting Percentage: Career")
plt.ylabel("Shooting Percentage: Pressure")




pressure["difference"] = pressure.percentage_pressure - pressure.percentage




pressure.sort_values(by="difference").head(10)




pressure.sort_values(by="difference", ascending=False).head(10)




df["shot_made_previous"] = df.shot_made.shift()

winner_effect = df[(df.play.str.contains("2 of 2")) | (df.play.str.contains("2 of 3")) | (df.play.str.contains("3 of 3"))]




made_previous = winner_effect[winner_effect.shot_made_previous==1].groupby("player")["shot_made"].agg(["mean", "size"])
made_previous = made_previous.rename(columns={"mean": "percentage_success", "size": "count_success"})
made_previous = made_previous[made_previous.count_success>=200]

made_previous = pd.merge(made_previous, shooting, left_index=True, right_index=True)
made_previous.head(3)




made_previous.plot(kind="scatter", x="percentage", y="percentage_success", figsize=(8,5))

worst_percentage = made_previous.percentage.min()
plt.plot([worst_percentage, 1], [worst_percentage, 1], color="red")

plt.title("Shooting Performance after Success", fontsize=20)
plt.xlabel("Shooting Percentage: Career")
plt.ylabel("Shooting Percentage: after Succes")




made_previous["difference"] = made_previous.percentage_success - made_previous.percentage




made_previous.sort_values(by="difference", ascending=False).head(10)




missed_previous = winner_effect[winner_effect.shot_made_previous==0].groupby("player")["shot_made"].agg(["mean", "size"])
missed_previous = missed_previous.rename(columns={"mean": "percentage_failure", "size": "count_failure"})
missed_previous = missed_previous[missed_previous.count_failure>=200]

missed_previous = pd.merge(missed_previous, shooting, left_index=True, right_index=True)
missed_previous.head(3)




missed_previous.plot(kind="scatter", x="percentage", y="percentage_failure", figsize=(8,5))

worst_percentage = missed_previous.percentage.min()
plt.plot([worst_percentage, 1], [worst_percentage, 1], color="red")

plt.title("Shooting Performance after Failure", fontsize=20)
plt.xlabel("Shooting Percentage: Career")
plt.ylabel("Shooting Percentage: after Failure")

