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




players = pd.read_csv('../input/players.csv') #match_id, hero_id
radiant_players = players[players.player_slot < 128]
radiant_players.head(11)




heroes = radiant_players.loc[:,['match_id','hero_id']]
heroes['is_used'] = pd.Series(np.ones(len(heroes['hero_id']), dtype=np.int8), index=heroes.index)
heroes.head(11)




usedHeroes = heroes.pivot_table(index='match_id', columns='hero_id', values='is_used', fill_value=0)
usedHeroes.head()




match = pd.read_csv('../input/match.csv') # radiant_win
anyWinHeroes = usedHeroes.merge(match.loc[:,['match_id','radiant_win']], left_index=True, right_on='match_id')
radiantWinHeroes = anyWinHeroes[anyWinHeroes.radiant_win == True]
del radiantWinHeroes['radiant_win']
del radiantWinHeroes['match_id']
radiantWinHeroes.head()




heroCorrelationMatrix = radiantWinHeroes.corr()
heroCorrelationMatrix.head()




heroCorrelation = heroCorrelationMatrix.stack().sort_values(ascending=False)
# heroCorrelation = heroCorrelation[heroCorrelation < 1.]
heroCorrelation[heroCorrelation < 1.].head(13)




dfHeroCorrelation = pd.DataFrame(heroCorrelation, index=heroCorrelation.index, columns=['corr'])
dfHeroCorrelation.index.rename(['h1', 'h2'], inplace=True)
dfHeroCorrelation.reset_index(inplace=True)
dfHeroCorrelation = dfHeroCorrelation[dfHeroCorrelation.h1 < dfHeroCorrelation.h2]
dfHeroCorrelation.head()




heroNames = pd.read_csv('../input/hero_names.csv')
heroNames = heroNames[['hero_id', 'localized_name']]
namedHeroCorrelation = dfHeroCorrelation.merge(heroNames, left_on='h1', right_on='hero_id')
namedHeroCorrelation = namedHeroCorrelation.merge(heroNames, left_on='h2', right_on='hero_id')
namedHeroCorrelation = namedHeroCorrelation[['corr', 'localized_name_x', 'localized_name_y']]
namedHeroCorrelation = namedHeroCorrelation.sort_values('corr', ascending=False)
namedHeroCorrelation.reset_index(drop=True, inplace=True)
namedHeroCorrelation[['localized_name_x', 'localized_name_y']]
# namedHeroCorrelation[namedHeroCorrelation.index % 2 == 0]




topPairs = dfHeroCorrelation.head(100)
heroesAndMatchesFromTopPairs = heroes.merge(
    heroes.merge(topPairs, left_on='hero_id', right_on='h1'),
    left_on=['match_id', 'hero_id'], right_on=['match_id', 'h2']
)
heroesAndMatchesFromTopPairs.head()




pairsWinLoseCounts = heroesAndMatchesFromTopPairs.merge(match[['match_id', 'radiant_win']], on='match_id')    [['match_id', 'h1', 'h2', 'radiant_win']].groupby(['h1', 'h2', 'radiant_win']).count()    .unstack()
pairsWinLoseCounts.head(11)




pairsWinLoseCounts['total_matches'] = (pairsWinLoseCounts['match_id'][True] + pairsWinLoseCounts['match_id'][False])
pairsWinLoseCounts['win_rate'] = pairsWinLoseCounts['match_id'][True] / pairsWinLoseCounts['total_matches']
del pairsWinLoseCounts['match_id']
pairsWinLoseCounts.reset_index(inplace=True)
pairsWinLoseCounts.columns = pairsWinLoseCounts.columns.droplevel(1)
pairsWinLoseCounts.head(11)




dfHeroCorrelationWithWinRate = dfHeroCorrelation.merge(pairsWinLoseCounts, on=['h1', 'h2'])
namedHeroCorrelation = dfHeroCorrelationWithWinRate.merge(heroNames, left_on='h1', right_on='hero_id')
namedHeroCorrelation = namedHeroCorrelation.merge(heroNames, left_on='h2', right_on='hero_id')
#namedHeroCorrelation = namedHeroCorrelation[['corr', 'localized_name_x', 'localized_name_y']]
namedHeroCorrelation = namedHeroCorrelation.sort_values('corr', ascending=False)
del namedHeroCorrelation['h1']
del namedHeroCorrelation['h2']
namedHeroCorrelation

