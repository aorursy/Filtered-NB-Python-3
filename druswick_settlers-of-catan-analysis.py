#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




games = pd.read_csv('../input/catanstats.csv')




# First, I figured we should look at the impact of starting position on game outcome. It looks like there may be some impact.




seaborn.boxplot(games['player'], games['points'], palette=seaborn.light_palette("purple")).set(xlabel='Placement Order', ylabel='Points', title='Settlers of Catan Placement Order vs Points', ylim=(0,14))




# Next, I decided to look at the correltion between certain endgame conditions and points. Robber cards seem to have an effect.




seaborn.regplot(games['robberCardsGain'], games['points'], color='purple').set(xlabel='Robber Card Gain', ylabel='Points', title='Robber Card Gain vs Points', ylim=(1,13))




# Card loss due to tribute seems not to matter too much. This could be that, for players who like to hold on to a lot of cards, the downside of losing cards is offset by the upside of being able to build easily and have plenty of stock to trade. 




games_trib = games[np.abs(games['tribute'] - np.mean(games['tribute'])) <= 3*games['tribute'].std()]
seaborn.regplot(games_trib['tribute'], games_trib['points'], color='purple').set(xlabel='Cards lost to Tribute', ylabel='Points', title='Tribute Loss vs Points', ylim=(1,13))




# And obviously, production is also major factor in points totals.




seaborn.regplot(games['production'], games['points'], color='purple').set(xlabel='Production', ylabel='Points', title='Production vs Points', ylim=(1,13))

