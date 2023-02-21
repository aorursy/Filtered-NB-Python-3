#!/usr/bin/env python
# coding: utf-8



# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

system("ls ../input")

# Any results you write to the current directory are saved as output.




library(ggplot2)
library(readr)
library(dplyr)
library(purrr)
library(tidyr)
library(ggplot2)
library(broom)
library(magrittr)
library(plotly)
library(RSQLite)
library(reshape2)
library(visNetwork)
library(networkD3)
library(jsonlite)
library(RColorBrewer)
library(gplots)
library(knitr)
library(DT)
library(data.table)
library(d3heatmap)
library(viridis)
library(maps)
library(ggmap)
library(circlize)




con <- dbConnect(SQLite(), dbname="../input/database.sqlite")

player       <- tbl_df(dbGetQuery(con,"SELECT * FROM player"))
player_attributes      <- tbl_df(dbGetQuery(con,"SELECT * FROM Player_Attributes"))

Match        <- tbl_df(dbGetQuery(con,"SELECT * FROM Match"))
Team        <- tbl_df(dbGetQuery(con,"SELECT * FROM Team"))
Country        <- tbl_df(dbGetQuery(con,"SELECT * FROM Country"))
League        <- tbl_df(dbGetQuery(con,"SELECT * FROM League"))

Zlatan_Attributes <- tbl_df(dbGetQuery(con,"SELECT * FROM Player_Attributes WHERE player_api_id = 35724"))
nrow(Zlatan_Attributes)




get_Lineup <- function(team_api_id, con) {

filter = "AND m.home_player_1 is NOT NULL AND m.home_player_2 is NOT NULL AND m.home_player_3 is NOT NULL AND m.home_player_4 is NOT NULL AND m.home_player_5 is NOT NULL AND m.home_player_6 is NOT NULL AND m.home_player_7 is NOT NULL AND m.home_player_8 is NOT NULL AND m.home_player_9 is NOT NULL AND m.home_player_10 is NOT NULL AND m.home_player_11 is NOT NULL"
p1 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p1,p.player_api_id as reference_number1 FROM player p, Match m WHERE home_team_api_id=",team_api_id,"AND p.player_api_id=m.home_player_1",filter)))
p2 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p2,p.player_api_id as reference_number2 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_2",filter)))
p3 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p3,p.player_api_id as reference_number3 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_3",filter)))
p4 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p4,p.player_api_id as reference_number4 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_4",filter)))
p5 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p5,p.player_api_id as reference_number5 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_5",filter)))
p6 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p6,p.player_api_id as reference_number6 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_6",filter)))
p7 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p7,p.player_api_id as reference_number7 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_7",filter)))
p8 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p8,p.player_api_id as reference_number8 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_8",filter)))
p9 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p9,p.player_api_id as reference_number9 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_9",filter)))
p10 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p10,p.player_api_id as reference_number10 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_10",filter)))
p11 <- tbl_df(dbGetQuery(con,paste("SELECT p.player_name as p11,p.player_api_id as reference_number11 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_11",filter)))

Lineup <- cbind(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11)
Lineup
}




#All Attributes Function

f_att <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height, m.weight, 
p.overall_rating,p.potential,p.crossing,p.heading_accuracy,p.short_passing,p.finishing, 
p.sprint_speed, p.long_shots, p.agility, p.vision, p.volleys, p.dribbling, p.curve, 
p.free_kick_accuracy, p.long_passing, p.ball_control, p.reactions, p.balance,
p.shot_power, p.jumping, p.stamina, p.strength, p.aggression, p.interceptions,
p.positioning, p.penalties, p.marking, p.standing_tackle, 
p.sliding_tackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

#removed: 'p.date, m.player_name' to create an exclusively numeric array
goalie_att <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT p.date, m.player_name, m.height, m.weight, 
p.overall_rating, p.reactions, p.balance, p.jumping, p.stamina, p.strength, p.aggression, p.interceptions,
p.positioning, p.gk_diving, p.gk_handling, p.gk_kicking, p.gk_positioning, gk_reflexes  FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}
##Critical Attributes Function - Forward




## English Premier League Lineups (TOP 6)
    #Manchester United Lineup
MU <- head(get_Lineup("10260", con))
    #Arsenal Lineup
ARS <-head(get_Lineup("9825", con))
    #Chelsea Lineup
CHS <-head(get_Lineup("8455", con))
    #Manchester City Lineup
MC <-head(get_Lineup("8456", con))
    #Fulham Lineup
FUL<-head(get_Lineup("9879", con))
    #Tottenham Lineup
TOT <-head(get_Lineup("8586", con))




head(get_Lineup("8455", con))




# Manchester United Attributes by match

#Goalie Attributes
for (id in MU$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in MU$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}

for (id in MU$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}

for (id in MU$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}

for (id in MU$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}

for (id in MU$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}

for (id in MU$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}

for (id in MU$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}

for (id in MU$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in MU$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in MU$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

MU_Lineup_Attributes <- cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




# Arsenal Attributes by match

#Goalie Attributes
for (id in ARS$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in ARS$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in ARS$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in ARS$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in ARS$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}

for (id in ARS$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in ARS$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in ARS$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in ARS$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in ARS$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in ARS$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}


ARS_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




CHS 




# Chelsea Attributes by match

#Goalie Attributes
for (id in CHS$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in CHS$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in CHS$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in CHS$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in CHS$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in CHS$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in CHS$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in CHS$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}

for (id in CHS$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in CHS$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in CHS$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

CHS_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)









# Manchester City Attributes by match

#Goalie Attributes
att_mean = c(3:18)
i = 3
for (id in MC$reference_number1){
    p1_att = head(goalie_att(id, con))
    att_mean[i] = colMeans(p1_att)
    i = i + 1
}
att_mean

for (id in MC$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in MC$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in MC$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in MC$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in MC$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in MC$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in MC$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in MC$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in MC$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in MC$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in MC$reference_number11){
    p11_att = head(f_att(id, con))
    
    p11_att
}


MC_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




att_mean = c()
i = 1
for (id in MC$reference_number1){
    p1_att = head(goalie_att(id, con))
    att_mean[i,1] = colMeans(p1_att[3:13])
    i = i + 1
}
att_mean




typeof(att_mean)




# Fulham Attributes by match

#Goalie Attributes
for (id in FUL$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in FUL$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in FUL$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in FUL$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in FUL$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in FUL$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in FUL$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in FUL$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in FUL$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in FUL$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in FUL$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

FUL_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




null




# Chelsea Attributes by match

#Goalie Attributes
for (id in TOT$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in TOT$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in TOT$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in TOT$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in TOT$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in TOT$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in TOT$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in TOT$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in TOT$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in TOT$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in TOT$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in TOT$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

TOT_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)





## Liga BBVA
    #Barcelona Lineup
BARC <-head(get_Lineup("8634", con))
    #Real Madrid Lineup
REAL<-head(get_Lineup("8633", con))
    #Sevilla Lineup
SEV<-head(get_Lineup("8302", con))
    #Atletico Madrid Lineup
ATL_M<-head(get_Lineup("9906", con))
    #Villareal Lineup
VIL<-head(get_Lineup("10205", con))
    #Atletico Bilbao Lineup
ATL_BILB<-head(get_Lineup("8315", con))




# Barca Attributes by match

#Goalie Attributes
for (id in BARC$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in BARC$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in BARC$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in BARC$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in BARC$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in BARC$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in BARC$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in BARC$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in BARC$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in BARC$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in BARC$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

BARC_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in REAL$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in REAL$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in REAL$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in REAL$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in REAL$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in REAL$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in REAL$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in REAL$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in REAL$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in REAL$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in REAL$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

REAL_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in ATL_M$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in ATL_M$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in ATL_M$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in ATL_M$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in ATL_M$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in ATL_M$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in ATL_M$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in ATL_M$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in ATL_M$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in ATL_M$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in ATL_M$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

ATL_M_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in SEV$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in SEV$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in SEV$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in SEV$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in SEV$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in SEV$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in SEV$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in SEV$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in SEV$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in SEV$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in SEV$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

SEV_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)





#Goalie Attributes
for (id in VIL$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in VIL$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in VIL$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in VIL$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in VIL$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in VIL$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in VIL$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in VIL$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in VIL$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in VIL$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in VIL$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

VIL_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)






#Goalie Attributes
for (id in ATL_BILB$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in ATL_BILB$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in ATL_BILB$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in ATL_BILB$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in ATL_BILB$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in ATL_BILB$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in ATL_BILB$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in ATL_BILB$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in ATL_BILB$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in ATL_BILB$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in ATL_BILB$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

ATL_BILB_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




##Serie A
    #Juventus Lineup
JUV<-head(get_Lineup("9885", con))
    #Milan Lineup
MIL<-head(get_Lineup("8564", con))
    #Roma Lineup
ROM<-head(get_Lineup("8686", con))
    #Inter Lineup
INT<-head(get_Lineup("8636", con))
    #Lazio Lineup
LAZ<-head(get_Lineup("8543", con))
    #Napoli Lineup
NAP<-head(get_Lineup("9875", con))




#Goalie Attributes
for (id in JUV$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in JUV$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in JUV$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in JUV$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in JUV$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in JUV$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in JUV$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in JUV$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in JUV$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in JUV$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in JUV$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

JUV_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)







#Goalie Attributes
for (id in MIL$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in MIL$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in MIL$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in MIL$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in MIL$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in MIL$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in MIL$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in MIL$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in MIL$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in MIL$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in MIL$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

MIL_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in ROM$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in ROM$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in ROM$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in ROM$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in ROM$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in ROM$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in ROM$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in ROM$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in ROM$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in ROM$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in ROM$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

ROM_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in INT$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in INT$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in INT$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in INT$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in INT$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in INT$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in INT$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in INT$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in INT$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in INT$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in INT$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

INT_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in LAZ$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in LAZ$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in LAZ$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in LAZ$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in LAZ$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in LAZ$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in LAZ$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in LAZ$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in LAZ$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in LAZ$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in LAZ$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

LAZ_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in NAP$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in NAP$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in NAP$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in NAP$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in NAP$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in NAP$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in NAP$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in NAP$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in NAP$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in NAP$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in NAP$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

NAP_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




##Bundesliga
    #Bayern Lineup
BFC<-head(get_Lineup("9823", con))
    #Borussia Dortmund Lineup
DORT<-head(get_Lineup("9789", con))
    #FC Schalke Lineup
SHLK<-head(get_Lineup("10189", con))
    #Wolfsburg Lineup
WOLF<-head(get_Lineup("8721", con))
    #Hannover Lineup
HAN<-head(get_Lineup("9904", con))
    #Bayern Leverkusen Lineup
BYL<-head(get_Lineup("8178", con))




#Goalie Attributes
for (id in BFC$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in BFC$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in BFC$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in BFC$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in BFC$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in BFC$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in BFC$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in BFC$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in BFC$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in BFC$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in BFC$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

BFC_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in DORT$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in DORT$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in DORT$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in DORT$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in DORT$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in DORT$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in DORT$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in DORT$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in DORT$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in DORT$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in DORT$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

DORT_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in SHLK$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in SHLK$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in SHLK$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in SHLK$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in SHLK$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in SHLK$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}

for (id in SHLK$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in SHLK$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in SHLK$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in SHLK$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in SHLK$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

SHLK_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)




#Goalie Attributes
for (id in BYL$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}

for (id in BYL$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}

for (id in BYL$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}

for (id in BYL$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in BYL$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in BYL$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in BYL$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in BYL$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in BYL$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in BYL$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in BYL$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}
BYL_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)





#Goalie Attributes
for (id in WOLF$reference_number1){
    p1_att = head(goalie_att(id, con))
    p1_att
}


for (id in WOLF$reference_number2){
    p2_att = head(f_att(id, con))
    p2_att
}


for (id in WOLF$reference_number3){
    p3_att = head(f_att(id, con))
    p3_att
}


for (id in WOLF$reference_number4){
    p4_att = head(f_att(id, con))
    p4_att
}


for (id in WOLF$reference_number5){
    p5_att = head(f_att(id, con))
    p5_att
}


for (id in WOLF$reference_number6){
    p6_att = head(f_att(id, con))
    p6_att
}



for (id in WOLF$reference_number7){
    p7_att = head(f_att(id, con))
    p7_att
}


for (id in WOLF$reference_number8){
    p8_att = head(f_att(id, con))
    p8_att
}


for (id in WOLF$reference_number9){
    p9_att = head(f_att(id, con))
    p9_att
}


for (id in WOLF$reference_number10){
    p10_att = head(f_att(id, con))
    p10_att
}


for (id in WOLF$reference_number11){
    p11_att = head(f_att(id, con))
    p11_att
}

WOLF_Lineup_Attributes<-cbind(p1_att, p2_att, p3_att, p4_att, p5_att, p6_att, p7_att, p8_att, p9_att, p10_att, p11_att)
summary(WOLF_Lineup_Attributes)




player  <- select(player,player_api_id, player_name) # use player_api_id as key for join
Team    <- select(Team, team_api_id, team_long_name, team_short_name) # use team_api_id as key for join
Country <-select(Country, id, name) %>% rename(country_id = id)  %>% rename(country_name = name)   # use country_id as key for join
League  <- select(League, country_id, name) %>% rename(league_name = name) # use country_id as key for join
Match   <-select(Match, id, country_id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11, goal, shoton, shotoff, foulcommit, card, cross, corner, possession)




player




MU_Lineup_Attributes
melted <- colMeans(MU_Lineup_Attributes)
melted


