#!/usr/bin/env python
# coding: utf-8



get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n\ndiv.h2 {\n    background-color: steelblue; \n    color: white; \n    padding: 8px; \n    padding-right: 300px; \n    font-size: 20px; \n    max-width: 1500px; \n    margin: auto; \n    margin-top: 50px;\n}\ndiv.h3 {\n    color: steelblue; \n    font-size: 14px; \n    margin-top: 20px; \n    margin-bottom:4px;\n}\ndiv.h4 {\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n}\nspan.note {\n    font-size: 5; \n    color: gray; \n    font-style: italic;\n}\nspan.captiona {\n    font-size: 5; \n    color: dimgray; \n    font-style: italic;\n    margin-left: 130px;\n    vertical-align: top;\n}\nhr {\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\nhr.light {\n    display: block; \n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n}\ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n    text-align: center;\n} \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n}\ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \ntable.rules tr.best\n{\n    color: green;\n}\n\n</style>')




# import
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import hvplot.pandas
from IPython.display import HTML, Image

# set additional display options for report
pd.set_option("display.max_columns", 100)
th_props = [('font-size', '13px'), ('background-color', 'white'), 
            ('color', '#666666')]
td_props = [('font-size', '15px'), ('background-color', 'white')]
styles = [dict(selector="td", props=td_props), dict(selector="th", 
            props=th_props)]




hist_df = pd.DataFrame({'Season': np.arange(2012,2018), 
            'Concussions': [265, 244, 212, 279, 250, 291]})
line_incidents = hist_df.hvplot.line(x='Season', y='Concussions', 
            xlim = (2011.5, 2017.5), ylim=(0,350), 
            title='Concussion Incidents for Full Season (incl Practice)',
            yticks=np.arange(50,350,50).tolist(), 
            xticks=np.arange(2012,2018).tolist(), grid=True)
scat_incidents = hist_df.hvplot.scatter(x='Season', y='Concussions'
            , size=50)
display(line_incidents * scat_incidents,
 HTML('<span class="captiona">' + 'Source: IQVIA' + '<span'))




years = [2016, 2017]
def get_sums(file, collist):
    df = pd.read_csv(file, usecols=['Year'] + collist)
    return [df.loc[df.Year == y, collist].sum().sum() for y in years]


scrims = get_sums('../input/nfl-competition-data/nflcom_scrims.csv', 
                  ['Scrm Plys'])
punts = get_sums('../input/nfl-competition-data/nflcom_punts.csv', 
                  ['Punts', 'Blk'])

punt_concussions = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv', 
                  usecols = ['Season_Year']).Season_Year.value_counts().values

kick_conc_pct = 0.12 #source: Kaggle competition Overview
scrimmage_concussions = np.array([217, 235]) * (1-kick_conc_pct) - punt_concussions 
                                                    

pcts_df = pd.DataFrame({'Passes_Runs': (scrimmage_concussions/scrims),
        'Punts': punt_concussions/punts}, index=years)
pcts_df['Risk_Multiple'] = (pcts_df.Punts/pcts_df.Passes_Runs).apply('{:.1f}'.format)
pcts_df['Passes_Runs'] =pcts_df.Passes_Runs.apply('{:.2%}'.format)
pcts_df['Punts'] =pcts_df.Punts.apply('{:.2%}'.format)
display(HTML('<span style="font-weight:bold">' + 'Concussion Percentages by Play Type'             + '</span>'),pcts_df) 




descriptions = pd.read_csv('../input/NFL-Punt-Analytics-Competition/play_information.csv', 
                        usecols=['PlayDescription'], nrows=50).PlayDescription.tolist()
for i in range(0,45,15):
    display(HTML('<span style="color:steelblue">' + descriptions[i] + '</span>'))




#%% get preprocessed play data
plays_all = pd.read_parquet('../input/nfl-data-preparation/plays.parq')
plays_all.set_index(['GameKey', 'PlayID'], inplace=True)


#%% parse text
outcomes = pd.DataFrame({'PlayDescription': plays_all.PlayDescription}, 
                                index=plays_all.index)
punttypes = {"not_punted":    ["no play", "delay of game", 
                               "false start", "blocked", "incomplete"],
             "out_of_bounds": ["out of bounds"], 
             "downed":        ["downed"],        
             "touchback":     ["touchback"],
             "fair_catch":    ["fair catch"],
             "returned":      ["no gain", "for (.*) yard"]
             }       
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for k,v in punttypes.items():
        outcomes[k] = outcomes.PlayDescription.str.contains('|'.join(v), 
                    case=False, regex=True).astype(int)


#%% correct for mulitple outcomes in one PlayID
outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[outcomes.PlayDescription.str.contains("punts", case=False, 
            regex=False), 'not_punted'] = 0
outcomes.loc[~outcomes.PlayDescription.str.contains("punts", case=False, 
            regex=False), 'out_of_bounds':'returned'] = 0
outcomes.loc[~outcomes.PlayDescription.str.contains("punts", case=False, 
            regex=False), 'not_punted'] = 1


outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[outcomes.typesum == 0, 'returned'] = 1
                        

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[(outcomes.PlayDescription.str.contains("invalid fair catch", 
            case=False, regex=False)) & (outcomes.returned == 1) & 
            (outcomes.typesum == 2), 'fair_catch'] = 0

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[(outcomes.PlayDescription.str.contains("punts", case=False, 
            regex=False)) & (outcomes.returned == 1) & (outcomes.typesum 
            == 2), 'not_punted':'out_of_bounds'] = 0

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[outcomes.PlayDescription.str.contains("blocked", case=False, 
            regex=False), 'out_of_bounds':'returned'] = 0

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[outcomes.typesum == 0, 'not_punted'] = 1 

outcomes['typesum'] = outcomes[list(punttypes.keys())].sum(axis=1)
outcomes.loc[(outcomes.touchback == 1) & (outcomes.typesum == 2), 
            'out_of_bounds':'downed'] = 0
outcomes.loc[(outcomes.returned == 1) & (outcomes.typesum == 2), 
            'returned'] = 0
outcomes.loc[(outcomes.fair_catch == 1) & (outcomes.typesum == 2), 
            'out_of_bounds':'downed'] = 0
outcomes.loc[(outcomes.downed == 1) & (outcomes.typesum == 2), 
            'out_of_bounds'] = 0

outcomes.drop(['PlayDescription', 'typesum'], axis=1, inplace=True)

plays_all['outcome'] = outcomes.dot(outcomes.columns).values #condense


#%% get yardage for return plays
plays_all['yardage'] = plays_all.PlayDescription.str            .extract("for (.{1,3}) yard")
plays_all.loc[plays_all.yardage.isnull(), 'yardage'] = 0
plays_all.loc[plays_all.outcome != "returned", 'yardage'] = 0




# format data for plotting
crosstable = pd.crosstab(plays_all.outcome, plays_all.concussion).reset_index()                    .sort_values(1, ascending=False)
crosstable.columns = ['Play_Outcome','Zero_Concussions', 'Concussions']
crosstable['Pct_of_Type'] = crosstable.Concussions/(crosstable.Concussions                + crosstable.Zero_Concussions)*100

bar_concs_all = crosstable.hvplot.bar('Play_Outcome', 'Concussions', ylim=(0,35), rot=45,
                yticks=np.arange(5,40,5).tolist(), width=400, height=300,
                  color="lightgray")
bar_concs_returned = crosstable[crosstable.Play_Outcome == 'returned'].hvplot                .bar('Play_Outcome', 'Concussions', title='Punt Concussions 2016-2017', 
                color='#ffa43d')

bar_pcts_all = crosstable.hvplot.bar('Play_Outcome', 'Pct_of_Type', ylim=(0,1.21), 
                rot=45, yticks=np.arange(0,1.4,0.2).tolist(), width=400, height=300,
                color="lightgray")
bar_pcts_returned = crosstable[crosstable.Play_Outcome == 'returned'].hvplot                .bar('Play_Outcome', 'Pct_of_Type', title='Punt Concussion Pcts 2016-2017', 
                color='#ffa43d')
                
display(bar_concs_all*bar_concs_returned + bar_pcts_all*bar_pcts_returned)
crosstable['Pct_of_Type'] = (crosstable.Pct_of_Type/100).apply('{:.2%}'.format)
crosstable['Play_Outcome'] = crosstable.Play_Outcome.str.title()
ctable = crosstable.sort_values('Concussions', ascending=False).set_index('Play_Outcome')

display(HTML('<span style="font-weight:bold">' + 'Comparison of Play Outcomes' + '</span>'), 
                ctable)




returns = plays_all.loc[(plays_all.outcome == "returned") & 
                        (~plays_all.PlayDescription.str.contains("MUFFS")), 
                        ['outcome', 'yardage', 'concussion', 'Rec_team']]\
                        .sort_values('Rec_team')
returns['yardage'] = returns.yardage.astype(int)
returns_median = returns.yardage.median()




import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(rc={'axes.facecolor':'darkseagreen'})
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mpl.rcParams['figure.dpi'] = 600
    mpl.rcParams["figure.figsize"] = [9, 6.5]
    mpl.rcParams['ytick.labelsize'] = 10 
    ax2 = sns.boxplot(x=returns.yardage, y = returns.Rec_team,
                      fliersize=4, whis=20, color="#CCCCCC")
    
    for i,artist in enumerate(ax2.artists):
        artist.set_edgecolor("lightgray")
        for j in range(i*6,i*6+6):
            line = ax2.lines[j]
            line.set_color("gray")
            line.set_mfc("lightgray")
            line.set_mec("lightgray")
    sns.stripplot(x=returns.yardage, y = returns.Rec_team, alpha=0.9, size=3, jitter=True, 
            color="steelblue")
    ax2.set_title('Punt Return Yardage', size=14, loc='left')
    ax2.set_title('2016-2017', size=14, loc='right')
    ax2.set_xlim([-15, 100])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    
    ax2.set_xlabel("Yards Gained", size=10)
    ax2.set_ylabel("Receiving Team", size=10)
    plt.axvline(returns_median, linewidth=1,  linestyle='-', color='red')




yardlist =  [0, 5, 10, 15, 20]
cutpts = [stats.percentileofscore(returns.yardage, yd)/100 for yd in yardlist]

pdf = pd.DataFrame({'Yards_Gained': yardlist, 'Total_Pct_of_Returns': cutpts})
pdf['Yards_Gained'] = pdf.Yards_Gained.astype(str).str.cat(["Yards or Less"]*len(yardlist), sep=" ")
pdf.set_index('Yards_Gained', inplace=True)
pdf['Total_Pct_of_Returns'] = pdf.Total_Pct_of_Returns.apply('{:.0%}'.format)

display(HTML('<span style="font-weight:bold">' + 'Punt Return Percentages by Yards Gained'             + '</span>'), pdf)




get_ipython().run_cell_magic('opts', 'Sankey [width=750]', 'import holoviews as hv\nfrom holoviews import opts\n\nconcussion_df = plays_all[plays_all.concussion == 1].copy()\n# activities = concussion_df.Player_Activity_Derived.value_counts().reset_index()\n# activities.columns = [\'Player_Activity\', \'Concussions\']\n# bar_activities = activities.hvplot.bar(\'Player_Activity\', \'Concussions\', ylim=(0,23), rot=0,\n#                yticks=np.arange(0,25,5).tolist(), width=400, height=300,\n#                color=\'lightgray\', title=\'Concussions by Player Activity\')\n\n# with warnings.catch_warnings():\n#     warnings.simplefilter("ignore")\n#     concussion_df.replace(\'unspecified\', \'Unclear\', inplace=True, regex=False)\n#     activity_combos = pd.crosstab(concussion_df.Player_Activity_Derived, \n#                     concussion_df.Primary_Partner_Activity_Derived)\n\n#     blocks = activity_combos.loc[\'Blocked\', \'Blocking\'] + activity_combos.loc[\'Blocking\', \n#                     \'Blocked\'] + activity_combos.loc[\'Blocked\', \'Blocked\']\n#     tackles = activity_combos.loc[\'Tackled\', \'Tackling\'] + activity_combos.loc[\'Tackling\',\n#                     \'Tackled\'] + activity_combos.loc[\'Tackling\', \'Tackling\']\n#     all_others = activity_combos.sum().sum() - tackles - blocks\n#     act_types = [\'Blocked_Blocking\', \'Tackled_Tackling\', \'All_Others\']\n#     acts_combo_df = pd.DataFrame({\'Combined_Activity\': act_types, \'Concussions\': [blocks, \n#                     tackles, all_others]})\n#     acts_combo_df.sort_values(\'Concussions\', ascending=False, inplace=True)\n\n    \n\n\nconcussion_df.Type_partner.replace("Unclear", "Unknown", inplace=True)\nconcussion_df.Type_player.replace("Unclear", "Unknown", inplace=True)\nconcussion_df[\'Player_Activity_Derived\'] = concussion_df.Player_Activity_Derived + [\'_\']\nconcussion_df[\'Type_player\'] = concussion_df.Type_player.str.title() + [\'_\']\nconcussion_df[\'Type_partner\'] = concussion_df.Type_partner.str.title()\n\n\ndef make_sankey(colfrom, colto):\n    conc_table = concussion_df[~concussion_df.Player_Activity_Derived.isnull()]\\\n                    .groupby([colfrom, colto])[\'PlayDescription\'].size()\\\n                    .to_frame().reset_index()\n    return conc_table.values\n\nsankey_cols = [\'Type_player\',\'Player_Activity_Derived\', \n                    \'Primary_Partner_Activity_Derived\', \'Type_partner\']\nsankey_list = []\nfor i in range(3):\n    sankey_piece = make_sankey(sankey_cols[i], sankey_cols[i+1])\n    sankey_list.append(sankey_piece)\n\nsankey_table = np.concatenate(sankey_list)\nc_sankey = hv.Sankey(sankey_table)\ndisplay(HTML(\'<span style="font-weight:bold; margin-left:84px">\' \\\n                 + \'Concussion Roles and Activities\' + \'</span>\'), c_sankey)')




from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

ngs = pd.read_parquet('../input/nfl-data-preparation/NGS.parq').sort_values(['GSISID', 
        'Time']).reset_index(drop=True)
ngs_piece = ngs[4281:4285].copy()
ngs_piece['x'] = ngs_piece.x.apply('{:.2f}'.format)
ngs_piece['y'] = ngs_piece.y.apply('{:.2f}'.format)
ngs_piece['dis'] = ngs_piece.dis.apply('{:.3f}'.format)
ngs_piece




get_ipython().run_cell_magic('HTML', '', '<a href="http://a.video.nfl.com//films/vodzilla/153321/Lechler_55_yd_punt-lG1K51rf-20181119_173634665_5000k.mp4"> \n    (2:57) (Punt formation) S.Lechler punts 48 yards to TEN 16, Center-J.Weeks. A.Jackson pushed ob at TEN 32 for 16 \n    yards (J.Jenkins).\n</a> \n<img src="https://s3.amazonaws.com/nonwebstorage/headstrong/animation_585_733_3.gif" width="650">')




x_dist = ngs.x - ngs.x.shift(-1), ngs.y - ngs.y.shift(-1)
ngs['speed'] = 10*np.hypot(ngs.x - ngs.x.shift(-1), ngs.y - ngs.y.shift(-1))

# get player-level agg
aggdict_player = {'Time': ['size'],
                  'x': ['mean', 'max', 'min', 'var'],
                  'y': ['mean', 'max', 'min', 'var'],
                  'speed': ['mean', 'max']}
ngs_agg = ngs.groupby(['GameKey', 'PlayID', 'GSISID']).agg(aggdict_player)
ngs_agg.columns = [n[0] + '_' + n[1] for n in ngs_agg.columns]
ngs_agg


# get play-level agg
aggdict_play = {'x_mean': ['mean', 'var'],
                'x_max': ['max', 'var'],
               'x_min': ['min'],
                'x_var': ['max', 'mean'],
                'y_mean': ['mean', 'var'],
                'y_max': ['max', 'var'],
               'y_min': ['min'],
                'y_var': ['max', 'mean'],
                'speed_mean': ['mean', 'var'],
                'speed_max': ['max', 'var']}
ngs_agg = ngs_agg.groupby(['GameKey', 'PlayID']).agg(aggdict_play)
ngs_agg.columns = [n[0] + '_' + n[1] for n in ngs_agg.columns]

plays_all['points_ahead'] = np.where(plays_all.Poss_Team == 
        plays_all.HomeTeamCode, plays_all.home_score - plays_all.visit_score,
        plays_all.visit_score - plays_all.home_score)

play_cols = ['Type_dlineman_agg',
             'Type_fullback_agg',
             'Type_gunner_agg',
             'Type_jammer_agg',
             'Type_linebacker_agg',
             'Type_olineman_agg',
             'Type_protector_agg',
             'Type_punter_agg', 
             'dist_togoal',
             'concussion',
             'yardage',
             'points_ahead',
             'PlayDescription']

plays_strategy = plays_all[play_cols]

ngs_agg2 =  ngs_agg.join(plays_strategy, how="inner").replace(-99, -1)
ngs_agg2['yardage'] = pd.to_numeric(ngs_agg2.yardage)

targets = ngs_agg2[['concussion', 'PlayDescription']]
ngs_agg2.drop(['concussion', 'PlayDescription'], axis=1, inplace=True)

ngs_scaled = StandardScaler().fit_transform(ngs_agg2.fillna(0).values)

tsne = TSNE(n_components=2, perplexity=30.0, verbose=1, 
        learning_rate=50, random_state=222)

ngs_emb = tsne.fit_transform(ngs_scaled)




strategy = pd.DataFrame(ngs_emb, columns=['x', 'y'])
strat_annotated = pd.concat([strategy, ngs_agg2.reset_index(), targets.reset_index(drop=True)], axis=1)

no_conc = strat_annotated[strat_annotated.concussion == 0].hvplot.scatter('x', 'y', 
                     alpha=0.4, size=5, grid=True, hover_cols=['PlayDescription'], 
                     height=550, width=700)
yes_conc = strat_annotated[strat_annotated.concussion == 1].hvplot.scatter('x', 'y', 
                     hover_cols=['PlayDescription'], size=30)
no_conc*yes_conc

