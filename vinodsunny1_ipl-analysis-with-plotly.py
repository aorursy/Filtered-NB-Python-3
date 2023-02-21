#!/usr/bin/env python
# coding: utf-8



from IPython.display import Image




Image("../input/images/t.jpg")




Image("../input/images/msd.jpg")




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
sns.set_style('darkgrid')
import warnings
import plotly.graph_objects as go
warnings.filterwarnings("ignore")




data = pd.read_csv("/kaggle/input/ipldata/matches.csv")
data = pd.DataFrame(data)
print("Original DataFrame Size :",data.shape)
data.head()




data.drop(["id","date","umpire1","umpire2","umpire3"],inplace = True,axis = 1)




missing_value = pd.DataFrame(data.isnull().sum(),columns =["counts"])
missing_value = missing_value[missing_value.counts > 0]
px.bar(data_frame = missing_value,x = missing_value.index ,y = "counts",color = ["blue","pink","orange"])




data.dropna(axis = 0,inplace = True)
print("Original DataFrame Size After removing  rows with atleast one null values  :",data.shape)




data.head(2)




data.winner      = data.winner.replace("Rising Pune Supergiant","Rising Pune Supergiants")
data.team1       = data.team1.replace("Rising Pune Supergiant","Rising Pune Supergiants")
data.team2       = data.team2.replace("Rising Pune Supergiant","Rising Pune Supergiants")
data.toss_winner = data.toss_winner.replace("Rising Pune Supergiant","Rising Pune Supergiants")




px.bar(x = data.winner.value_counts().index, y = data.winner.value_counts().values , color =data.winner.value_counts().index,labels = {"x":"IPL Teams","y":"Number of Mactch's won By Indvivual Team"})




px.bar(x = data.toss_winner.value_counts().values,y = data.toss_winner.value_counts().index,orientation = "h",color = data.toss_winner.value_counts().index ,labels = {"y":"Teams","x":"Number of Tosse's won By Indvivual Team"})




px.pie(names = data.toss_decision.value_counts().index ,values =  data.toss_decision.value_counts().values,title = "Fielding or Batting")




data[data.result == "tie"][["season","team1","team2","result"]]




px.pie(names = data.city.value_counts().index,values = data.city.value_counts().values,title = "Matches Held in Different Cities")




names  = data.player_of_match.value_counts().head(20).index
values = data.player_of_match.value_counts().head(20).values
color  = data.player_of_match.value_counts().head(20).index
px.bar(y = names,x = values,color = color,orientation='h',labels ={"y":"Player's","x":"Number of Man of The Match Awards"})




won_by_more_then_hun = data[data.win_by_runs > 100][["winner","team1","team2","win_by_runs"]]  
won_by_more_then_hun.head(10)




values = won_by_more_then_hun.winner.value_counts().values
names  = won_by_more_then_hun.winner.value_counts().index 
px.bar(x = names,y = values,color = names,labels = {"x":"Teams won By More Than 100 runs ","y":"Number Of time teams won by more than 100 runs"})




data.season.value_counts()
fig = px.pie(names = data.season.value_counts().index, values = data.season.value_counts().values,title = "Number Of Matches were played  in each of the season's",hole = 0.5)
fig.update_traces(textinfo='value')
fig.show()




won_by_7wicks = data[data.win_by_wickets >7][["winner","team1","team2","win_by_wickets"]]  
won_by_7wicks




values = won_by_7wicks.winner.value_counts().values
names  = won_by_7wicks.winner.value_counts().index 
px.bar(x = names,y = values,color = names,labels = {"x":"Teams won By Saving More Than 7 Wicket's ","y":"Number Of time teams won by Saving More than 7 wickets"})




plt.rcParams["figure.figsize"] = (15,15)
sns.barplot(y = data.venue.value_counts().index ,x = data.venue.value_counts().values,orient = "h",palette = 'CMRmap')
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.show()




Image("../input/images/d2.jpg")




# Here Comes Captain cool !
msd_data = data[data.player_of_match =="MS Dhoni"]
msd_data.winner.value_counts()




print("Total Number Of Man Of the Match awards won By MSD : ",msd_data.shape[0])
print("Sum Of Matches own by CSK and RPS When MSD Got MAN of the Match Award: ",(msd_data.winner.value_counts()[0]+msd_data.winner.value_counts()[1]))




fig = px.pie(names = msd_data.season.value_counts().index, values = msd_data.season.value_counts().values,title = "Number Of Man of the Match Awards Own by MSD in Each Seasons",hole= 0.5)
fig.update_traces(textinfo='value')
fig.show()




Image("../input/images/cr.jpg")




CSK_RCB = data[((data.team1 =="Royal Challengers Bangalore") & (data.team2 =="Chennai Super Kings")) | ((data.team2 =="Royal Challengers Bangalore") & (data.team1 =="Chennai Super Kings"))]
print("Total Number Of matches Held between CSK and RCB : ",CSK_RCB.shape[0])
CSK_RCB.head(3)




name  = CSK_RCB.winner.value_counts().index
value = CSK_RCB.winner.value_counts().values
px.pie(names = name,values = value)




name  = CSK_RCB.toss_winner.value_counts().index
value = CSK_RCB.toss_winner.value_counts().values
px.bar(x = name,y = value,color = name,labels = {"x":"CSK v/s RCB ","y":"Number Of time toss's won "})




CSK_RCB.player_of_match.value_counts().head(5)




Image("../input/images/sr1.png")




name  = CSK_RCB.player_of_match.value_counts().head(5).index
value = CSK_RCB.player_of_match.value_counts().head(5).values
px.bar(x = name,y = value,color = name,labels = {"x":"Number of times Man of The Match Award","y":"Player's From csk and rcb "})




print("Toss Decision ")
x = pd.crosstab(CSK_RCB['toss_winner'], CSK_RCB['toss_decision'])
x = pd.DataFrame(x)
x




plt.rcParams["figure.figsize"] = (10,5)
CSK_RCB.groupby(["toss_winner","toss_decision"])["winner"].count().plot(kind = 'bar')
plt.xticks(rotation = 60)
plt.show()






