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
library(plyr)
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
#library(ggmap)
library(circlize)





con <- dbConnect(SQLite(), dbname="../input/database.sqlite")
###
Player       <- tbl_df(dbGetQuery(con,"SELECT * FROM player"))
Player_Attributes      <- tbl_df(dbGetQuery(con,"SELECT * FROM Player_Attributes"))
Match        <- tbl_df(dbGetQuery(con,"SELECT * FROM Match"))
Team        <- tbl_df(dbGetQuery(con,"SELECT * FROM Team"))
Country        <- tbl_df(dbGetQuery(con,"SELECT * FROM Country"))
League        <- tbl_df(dbGetQuery(con,"SELECT * FROM League"))

###


get_Lineup <- function(team_api_id, con) {

filter = "AND m.home_player_1 is NOT NULL AND m.home_player_2 is NOT NULL AND m.home_player_3 is NOT NULL AND m.home_player_4 is NOT NULL AND m.home_player_5 is NOT NULL AND m.home_player_6 is NOT NULL AND m.home_player_7 is NOT NULL AND m.home_player_8 is NOT NULL AND m.home_player_9 is NOT NULL AND m.home_player_10 is NOT NULL AND m.home_player_11 is NOT NULL"
p1 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p1,p.player_api_id as reference_number1 FROM player p, Match m WHERE home_team_api_id=",team_api_id,"AND p.player_api_id=m.home_player_1",filter)))
p2 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p2,p.player_api_id as reference_number2 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_2",filter)))
p3 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p3,p.player_api_id as reference_number3 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_3",filter)))
p4 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p4,p.player_api_id as reference_number4 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_4",filter)))
p5 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p5,p.player_api_id as reference_number5 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_5",filter)))
p6 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p6,p.player_api_id as reference_number6 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_6",filter)))
p7 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p7,p.player_api_id as reference_number7 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_7",filter)))
p8 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p8,p.player_api_id as reference_number8 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_8",filter)))
p9 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p9,p.player_api_id as reference_number9 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_9",filter)))
p10 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p10,p.player_api_id as reference_number10 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_10",filter)))
p11 <- tbl_df(dbGetQuery(con,paste("SELECT m.home_team_api_id,p.player_name as p11,p.player_api_id as reference_number11 FROM player p, Match m WHERE home_team_api_id=",team_api_id," AND p.player_api_id=m.home_player_11",filter)))

Lineup <- cbind(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11)
Lineup <- head(Lineup, n=10)
Lineup
}




##team1 attributes
###
#All Attributes Function

#removed: 'p.date, m.player_name' to create an exclusively numeric array



team_1goalie_att <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT p.date, m.player_name, m.height as team_1g_height, m.weight as team_1g_weight, 
p.overall_rating as team_1g_overall, p.potential as team_1g_potential, p.gk_diving as team_1g_diving, p.gk_handling as team_1g_handling, p.gk_kicking as team_1g_kicking, p.gk_positioning as team_1g_positioning, gk_reflexes as team_1g_reflexes FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}


###
#All Attributes Function

team_1f_att2 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p2_weight, 
p.overall_rating as team_1p2_overall,p.potential as team_1p2_potential,p.crossing as team_1p2_crossing,p.heading_accuracy as team_1p2_heading,p.short_passing as team_1p2_shortpass,p.finishing as team_1p2_finishing, 
p.sprint_speed as team_1p2_sprintspeed, p.long_shots as team_1p2_longshots, p.agility as team_1p2_agility, p.vision as team_1p2_vision, p.volleys as team_1p2_volleys, p.dribbling as team_1p2_dribbling, p.curve as team_1p2_curve, 
p.free_kick_accuracy as team_1p2_freekickacc, p.long_passing as team_1p2_longpassing, p.ball_control as team_1p2_ballcontrol, p.reactions as team_1p2_reactions, p.balance as team_1p2_balance,
p.shot_power as team_1p2_shotpower, p.jumping as team_1p2_jumping, p.stamina as team_1p2_stamina, p.strength as team_1p2_strength, p.aggression as team_1p2_aggression, p.interceptions as team_1p2_interceptions,
p.positioning as team_1p2_positioning, p.penalties as team_1p2_penalties, p.marking as team_1p2_marking, p.standing_tackle as team_1p2_standingtackle, 
p.sliding_tackle as team_1p2_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att3 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p3_weight, 
p.overall_rating as team_1p3_overall,p.potential as team_1p3_potential,p.crossing as team_1p3_crossing,p.heading_accuracy as team_1p3_heading,p.short_passing as team_1p3_shortpass,p.finishing as team_1p3_finishing, 
p.sprint_speed as team_1p3_sprintspeed, p.long_shots as team_1p3_longshots, p.agility as team_1p3_agility, p.vision as team_1p3_vision, p.volleys as team_1p3_volleys, p.dribbling as team_1p3_dribbling, p.curve as team_1p3_curve, 
p.free_kick_accuracy as team_1p3_freekickacc, p.long_passing as team_1p3_longpassing, p.ball_control as team_1p3_ballcontrol, p.reactions as team_1p3_reactions, p.balance as team_1p3_balance,
p.shot_power as team_1p3_shotpower, p.jumping as team_1p3_jumping, p.stamina as team_1p3_stamina, p.strength as team_1p3_strength, p.aggression as team_1p3_aggression, p.interceptions as team_1p3_interceptions,
p.positioning as team_1p3_positioning, p.penalties as team_1p3_penalties, p.marking as team_1p3_marking, p.standing_tackle as team_1p3_standingtackle, 
p.sliding_tackle as team_1p3_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att4 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p4_weight, 
p.overall_rating as team_1p4_overall,p.potential as team_1p4_potential,p.crossing as team_1p4_crossing,p.heading_accuracy as team_1p4_heading,p.short_passing as team_1p4_shortpass,p.finishing as team_1p4_finishing, 
p.sprint_speed as team_1p4_sprintspeed, p.long_shots as team_1p4_longshots, p.agility as team_1p4_agility, p.vision as team_1p4_vision, p.volleys as team_1p4_volleys, p.dribbling as team_1p4_dribbling, p.curve as team_1p4_curve, 
p.free_kick_accuracy as team_1p4_freekickacc, p.long_passing as team_1p4_longpassing, p.ball_control as team_1p4_ballcontrol, p.reactions as team_1p4_reactions, p.balance as team_1p4_balance,
p.shot_power as team_1p4_shotpower, p.jumping as team_1p4_jumping, p.stamina as team_1p4_stamina, p.strength as team_1p4_strength, p.aggression as team_1p4_aggression, p.interceptions as team_1p4_interceptions,
p.positioning as team_1p4_positioning, p.penalties as team_1p4_penalties, p.marking as team_1p4_marking, p.standing_tackle as team_1p4_standingtackle, 
p.sliding_tackle as team_1p4_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att5 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p5_weight, 
p.overall_rating as team_1p5_overall,p.potential as team_1p5_potential,p.crossing as team_1p5_crossing,p.heading_accuracy as team_1p5_heading,p.short_passing as team_1p5_shortpass,p.finishing as team_1p5_finishing, 
p.sprint_speed as team_1p5_sprintspeed, p.long_shots as team_1p5_longshots, p.agility as team_1p5_agility, p.vision as team_1p5_vision, p.volleys as team_1p5_volleys, p.dribbling as team_1p5_dribbling, p.curve as team_1p5_curve, 
p.free_kick_accuracy as team_1p5_freekickacc, p.long_passing as team_1p5_longpassing, p.ball_control as team_1p5_ballcontrol, p.reactions as team_1p5_reactions, p.balance as team_1p5_balance,
p.shot_power as team_1p5_shotpower, p.jumping as team_1p5_jumping, p.stamina as team_1p5_stamina, p.strength as team_1p5_strength, p.aggression as team_1p5_aggression, p.interceptions as team_1p5_interceptions,
p.positioning as team_1p5_positioning, p.penalties as team_1p5_penalties, p.marking as team_1p5_marking, p.standing_tackle as team_1p5_standingtackle, 
p.sliding_tackle as team_1p5_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att6 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p6_weight, 
p.overall_rating as team_1p6_overall,p.potential as team_1p6_potential,p.crossing as team_1p6_crossing,p.heading_accuracy as team_1p6_heading,p.short_passing as team_1p6_shortpass,p.finishing as team_1p6_finishing, 
p.sprint_speed as team_1p6_sprintspeed, p.long_shots as team_1p6_longshots, p.agility as team_1p6_agility, p.vision as team_1p6_vision, p.volleys as team_1p6_volleys, p.dribbling as team_1p6_dribbling, p.curve as team_1p6_curve, 
p.free_kick_accuracy as team_1p6_freekickacc, p.long_passing as team_1p6_longpassing, p.ball_control as team_1p6_ballcontrol, p.reactions as team_1p6_reactions, p.balance as team_1p6_balance,
p.shot_power as team_1p6_shotpower, p.jumping as team_1p6_jumping, p.stamina as team_1p6_stamina, p.strength as team_1p6_strength, p.aggression as team_1p6_aggression, p.interceptions as team_1p6_interceptions,
p.positioning as team_1p6_positioning, p.penalties as team_1p6_penalties, p.marking as team_1p6_marking, p.standing_tackle as team_1p6_standingtackle, 
p.sliding_tackle as team_1p6_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att7 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p7_weight, 
p.overall_rating as team_1p7_overall,p.potential as team_1p7_potential,p.crossing as team_1p7_crossing,p.heading_accuracy as team_1p7_heading,p.short_passing as team_1p7_shortpass,p.finishing as team_1p7_finishing, 
p.sprint_speed as team_1p7_sprintspeed, p.long_shots as team_1p7_longshots, p.agility as team_1p7_agility, p.vision as team_1p7_vision, p.volleys as team_1p7_volleys, p.dribbling as team_1p7_dribbling, p.curve as team_1p7_curve, 
p.free_kick_accuracy as team_1p7_freekickacc, p.long_passing as team_1p7_longpassing, p.ball_control as team_1p7_ballcontrol, p.reactions as team_1p7_reactions, p.balance as team_1p7_balance,
p.shot_power as team_1p7_shotpower, p.jumping as team_1p7_jumping, p.stamina as team_1p7_stamina, p.strength as team_1p7_strength, p.aggression as team_1p7_aggression, p.interceptions as team_1p7_interceptions,
p.positioning as team_1p7_positioning, p.penalties as team_1p7_penalties, p.marking as team_1p7_marking, p.standing_tackle as team_1p7_standingtackle, 
p.sliding_tackle as team_1p7_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att8 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p8_weight, 
p.overall_rating as team_1p8_overall,p.potential as team_1p8_potential,p.crossing as team_1p8_crossing,p.heading_accuracy as team_1p8_heading,p.short_passing as team_1p8_shortpass,p.finishing as team_1p8_finishing, 
p.sprint_speed as team_1p8_sprintspeed, p.long_shots as team_1p8_longshots, p.agility as team_1p8_agility, p.vision as team_1p8_vision, p.volleys as team_1p8_volleys, p.dribbling as team_1p8_dribbling, p.curve as team_1p8_curve, 
p.free_kick_accuracy as team_1p8_freekickacc, p.long_passing as team_1p8_longpassing, p.ball_control as team_1p8_ballcontrol, p.reactions as team_1p8_reactions, p.balance as team_1p8_balance,
p.shot_power as team_1p8_shotpower, p.jumping as team_1p8_jumping, p.stamina as team_1p8_stamina, p.strength as team_1p8_strength, p.aggression as team_1p8_aggression, p.interceptions as team_1p8_interceptions,
p.positioning as team_1p8_positioning, p.penalties as team_1p8_penalties, p.marking as team_1p8_marking, p.standing_tackle as team_1p8_standingtackle, 
p.sliding_tackle as team_1p8_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att9 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p9_weight, 
p.overall_rating as team_1p9_overall,p.potential as team_1p9_potential,p.crossing as team_1p9_crossing,p.heading_accuracy as team_1p9_heading,p.short_passing as team_1p9_shortpass,p.finishing as team_1p9_finishing, 
p.sprint_speed as team_1p9_sprintspeed, p.long_shots as team_1p9_longshots, p.agility as team_1p9_agility, p.vision as team_1p9_vision, p.volleys as team_1p9_volleys, p.dribbling as team_1p9_dribbling, p.curve as team_1p9_curve, 
p.free_kick_accuracy as team_1p9_freekickacc, p.long_passing as team_1p9_longpassing, p.ball_control as team_1p9_ballcontrol, p.reactions as team_1p9_reactions, p.balance as team_1p9_balance,
p.shot_power as team_1p9_shotpower, p.jumping as team_1p9_jumping, p.stamina as team_1p9_stamina, p.strength as team_1p9_strength, p.aggression as team_1p9_aggression, p.interceptions as team_1p9_interceptions,
p.positioning as team_1p9_positioning, p.penalties as team_1p9_penalties, p.marking as team_1p9_marking, p.standing_tackle as team_1p9_standingtackle, 
p.sliding_tackle as team_1p9_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att10 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p10_weight, 
p.overall_rating as team_1p10_overall,p.potential as team_1p10_potential,p.crossing as team_1p10_crossing,p.heading_accuracy as team_1p10_heading,p.short_passing as team_1p10_shortpass,p.finishing as team_1p10_finishing, 
p.sprint_speed as team_1p10_sprintspeed, p.long_shots as team_1p10_longshots, p.agility as team_1p10_agility, p.vision as team_1p10_vision, p.volleys as team_1p10_volleys, p.dribbling as team_1p10_dribbling, p.curve as team_1p10_curve, 
p.free_kick_accuracy as team_1p10_freekickacc, p.long_passing as team_1p10_longpassing, p.ball_control as team_1p10_ballcontrol, p.reactions as team_1p10_reactions, p.balance as team_1p10_balance,
p.shot_power as team_1p10_shotpower, p.jumping as team_1p10_jumping, p.stamina as team_1p10_stamina, p.strength as team_1p10_strength, p.aggression as team_1p10_aggression, p.interceptions as team_1p10_interceptions,
p.positioning as team_1p10_positioning, p.penalties as team_1p10_penalties, p.marking as team_1p10_marking, p.standing_tackle as team_1p10_standingtackle, 
p.sliding_tackle as team_1p10_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_1f_att11 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_1p11_weight, 
p.overall_rating as team_1p11_overall,p.potential as team_1p11_potential,p.crossing as team_1p11_crossing,p.heading_accuracy as team_1p11_heading,p.short_passing as team_1p11_shortpass,p.finishing as team_1p11_finishing, 
p.sprint_speed as team_1p11_sprintspeed, p.long_shots as team_1p11_longshots, p.agility as team_1p11_agility, p.vision as team_1p11_vision, p.volleys as team_1p11_volleys, p.dribbling as team_1p11_dribbling, p.curve as team_1p11_curve, 
p.free_kick_accuracy as team_1p11_freekickacc, p.long_passing as team_1p11_longpassing, p.ball_control as team_1p11_ballcontrol, p.reactions as team_1p11_reactions, p.balance as team_1p11_balance,
p.shot_power as team_1p11_shotpower, p.jumping as team_1p11_jumping, p.stamina as team_1p11_stamina, p.strength as team_1p11_strength, p.aggression as team_1p11_aggression, p.interceptions as team_1p11_interceptions,
p.positioning as team_1p11_positioning, p.penalties as team_1p11_penalties, p.marking as team_1p11_marking, p.standing_tackle as team_1p11_standingtackle, 
p.sliding_tackle as team_1p11_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}
#removed: 'p.date, m.player_name' to create an exclusively numeric array





##team2 attributes
team_2goalie_att <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT p.date, m.player_name, m.height as team_2g_height, m.weight as team_2g_weight, 
p.overall_rating as team_2g_overall, p.potential as team_2g_potential, p.gk_diving as team_2g_diving, p.gk_handling as team_2g_handling, p.gk_kicking as team_2g_kicking, p.gk_positioning as team_2g_positioning, gk_reflexes as team_2g_reflexes FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}


###
#All Attributes Function

team_2f_att2 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p2_weight, 
p.overall_rating as team_2p2_overall,p.potential as team_2p2_potential,p.crossing as team_2p2_crossing,p.heading_accuracy as team_2p2_heading,p.short_passing as team_2p2_shortpass,p.finishing as team_2p2_finishing, 
p.sprint_speed as team_2p2_sprintspeed, p.long_shots as team_2p2_longshots, p.agility as team_2p2_agility, p.vision as team_2p2_vision, p.volleys as team_2p2_volleys, p.dribbling as team_2p2_dribbling, p.curve as team_2p2_curve, 
p.free_kick_accuracy as team_2p2_freekickacc, p.long_passing as team_2p2_longpassing, p.ball_control as team_2p2_ballcontrol, p.reactions as team_2p2_reactions, p.balance as team_2p2_balance,
p.shot_power as team_2p2_shotpower, p.jumping as team_2p2_jumping, p.stamina as team_2p2_stamina, p.strength as team_2p2_strength, p.aggression as team_2p2_aggression, p.interceptions as team_2p2_interceptions,
p.positioning as team_2p2_positioning, p.penalties as team_2p2_penalties, p.marking as team_2p2_marking, p.standing_tackle as team_2p2_standingtackle, 
p.sliding_tackle as team_2p2_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att3 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p3_weight, 
p.overall_rating as team_2p3_overall,p.potential as team_2p3_potential,p.crossing as team_2p3_crossing,p.heading_accuracy as team_2p3_heading,p.short_passing as team_2p3_shortpass,p.finishing as team_2p3_finishing, 
p.sprint_speed as team_2p3_sprintspeed, p.long_shots as team_2p3_longshots, p.agility as team_2p3_agility, p.vision as team_2p3_vision, p.volleys as team_2p3_volleys, p.dribbling as team_2p3_dribbling, p.curve as team_2p3_curve, 
p.free_kick_accuracy as team_2p3_freekickacc, p.long_passing as team_2p3_longpassing, p.ball_control as team_2p3_ballcontrol, p.reactions as team_2p3_reactions, p.balance as team_2p3_balance,
p.shot_power as team_2p3_shotpower, p.jumping as team_2p3_jumping, p.stamina as team_2p3_stamina, p.strength as team_2p3_strength, p.aggression as team_2p3_aggression, p.interceptions as team_2p3_interceptions,
p.positioning as team_2p3_positioning, p.penalties as team_2p3_penalties, p.marking as team_2p3_marking, p.standing_tackle as team_2p3_standingtackle, 
p.sliding_tackle as team_2p3_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att4 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p4_weight, 
p.overall_rating as team_2p4_overall,p.potential as team_2p4_potential,p.crossing as team_2p4_crossing,p.heading_accuracy as team_2p4_heading,p.short_passing as team_2p4_shortpass,p.finishing as team_2p4_finishing, 
p.sprint_speed as team_2p4_sprintspeed, p.long_shots as team_2p4_longshots, p.agility as team_2p4_agility, p.vision as team_2p4_vision, p.volleys as team_2p4_volleys, p.dribbling as team_2p4_dribbling, p.curve as team_2p4_curve, 
p.free_kick_accuracy as team_2p4_freekickacc, p.long_passing as team_2p4_longpassing, p.ball_control as team_2p4_ballcontrol, p.reactions as team_2p4_reactions, p.balance as team_2p4_balance,
p.shot_power as team_2p4_shotpower, p.jumping as team_2p4_jumping, p.stamina as team_2p4_stamina, p.strength as team_2p4_strength, p.aggression as team_2p4_aggression, p.interceptions as team_2p4_interceptions,
p.positioning as team_2p4_positioning, p.penalties as team_2p4_penalties, p.marking as team_2p4_marking, p.standing_tackle as team_2p4_standingtackle, 
p.sliding_tackle as team_2p4_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att5 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p5_weight, 
p.overall_rating as team_2p5_overall,p.potential as team_2p5_potential,p.crossing as team_2p5_crossing,p.heading_accuracy as team_2p5_heading,p.short_passing as team_2p5_shortpass,p.finishing as team_2p5_finishing, 
p.sprint_speed as team_2p5_sprintspeed, p.long_shots as team_2p5_longshots, p.agility as team_2p5_agility, p.vision as team_2p5_vision, p.volleys as team_2p5_volleys, p.dribbling as team_2p5_dribbling, p.curve as team_2p5_curve, 
p.free_kick_accuracy as team_2p5_freekickacc, p.long_passing as team_2p5_longpassing, p.ball_control as team_2p5_ballcontrol, p.reactions as team_2p5_reactions, p.balance as team_2p5_balance,
p.shot_power as team_2p5_shotpower, p.jumping as team_2p5_jumping, p.stamina as team_2p5_stamina, p.strength as team_2p5_strength, p.aggression as team_2p5_aggression, p.interceptions as team_2p5_interceptions,
p.positioning as team_2p5_positioning, p.penalties as team_2p5_penalties, p.marking as team_2p5_marking, p.standing_tackle as team_2p5_standingtackle, 
p.sliding_tackle as team_2p5_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att6 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p6_weight, 
p.overall_rating as team_2p6_overall,p.potential as team_2p6_potential,p.crossing as team_2p6_crossing,p.heading_accuracy as team_2p6_heading,p.short_passing as team_2p6_shortpass,p.finishing as team_2p6_finishing, 
p.sprint_speed as team_2p6_sprintspeed, p.long_shots as team_2p6_longshots, p.agility as team_2p6_agility, p.vision as team_2p6_vision, p.volleys as team_2p6_volleys, p.dribbling as team_2p6_dribbling, p.curve as team_2p6_curve, 
p.free_kick_accuracy as team_2p6_freekickacc, p.long_passing as team_2p6_longpassing, p.ball_control as team_2p6_ballcontrol, p.reactions as team_2p6_reactions, p.balance as team_2p6_balance,
p.shot_power as team_2p6_shotpower, p.jumping as team_2p6_jumping, p.stamina as team_2p6_stamina, p.strength as team_2p6_strength, p.aggression as team_2p6_aggression, p.interceptions as team_2p6_interceptions,
p.positioning as team_2p6_positioning, p.penalties as team_2p6_penalties, p.marking as team_2p6_marking, p.standing_tackle as team_2p6_standingtackle, 
p.sliding_tackle as team_2p6_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att7 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p7_weight, 
p.overall_rating as team_2p7_overall,p.potential as team_2p7_potential,p.crossing as team_2p7_crossing,p.heading_accuracy as team_2p7_heading,p.short_passing as team_2p7_shortpass,p.finishing as team_2p7_finishing, 
p.sprint_speed as team_2p7_sprintspeed, p.long_shots as team_2p7_longshots, p.agility as team_2p7_agility, p.vision as team_2p7_vision, p.volleys as team_2p7_volleys, p.dribbling as team_2p7_dribbling, p.curve as team_2p7_curve, 
p.free_kick_accuracy as team_2p7_freekickacc, p.long_passing as team_2p7_longpassing, p.ball_control as team_2p7_ballcontrol, p.reactions as team_2p7_reactions, p.balance as team_2p7_balance,
p.shot_power as team_2p7_shotpower, p.jumping as team_2p7_jumping, p.stamina as team_2p7_stamina, p.strength as team_2p7_strength, p.aggression as team_2p7_aggression, p.interceptions as team_2p7_interceptions,
p.positioning as team_2p7_positioning, p.penalties as team_2p7_penalties, p.marking as team_2p7_marking, p.standing_tackle as team_2p7_standingtackle, 
p.sliding_tackle as team_2p7_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att8 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p8_weight, 
p.overall_rating as team_2p8_overall,p.potential as team_2p8_potential,p.crossing as team_2p8_crossing,p.heading_accuracy as team_2p8_heading,p.short_passing as team_2p8_shortpass,p.finishing as team_2p8_finishing, 
p.sprint_speed as team_2p8_sprintspeed, p.long_shots as team_2p8_longshots, p.agility as team_2p8_agility, p.vision as team_2p8_vision, p.volleys as team_2p8_volleys, p.dribbling as team_2p8_dribbling, p.curve as team_2p8_curve, 
p.free_kick_accuracy as team_2p8_freekickacc, p.long_passing as team_2p8_longpassing, p.ball_control as team_2p8_ballcontrol, p.reactions as team_2p8_reactions, p.balance as team_2p8_balance,
p.shot_power as team_2p8_shotpower, p.jumping as team_2p8_jumping, p.stamina as team_2p8_stamina, p.strength as team_2p8_strength, p.aggression as team_2p8_aggression, p.interceptions as team_2p8_interceptions,
p.positioning as team_2p8_positioning, p.penalties as team_2p8_penalties, p.marking as team_2p8_marking, p.standing_tackle as team_2p8_standingtackle, 
p.sliding_tackle as team_2p8_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att9 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p9_weight, 
p.overall_rating as team_2p9_overall,p.potential as team_2p9_potential,p.crossing as team_2p9_crossing,p.heading_accuracy as team_2p9_heading,p.short_passing as team_2p9_shortpass,p.finishing as team_2p9_finishing, 
p.sprint_speed as team_2p9_sprintspeed, p.long_shots as team_2p9_longshots, p.agility as team_2p9_agility, p.vision as team_2p9_vision, p.volleys as team_2p9_volleys, p.dribbling as team_2p9_dribbling, p.curve as team_2p9_curve, 
p.free_kick_accuracy as team_2p9_freekickacc, p.long_passing as team_2p9_longpassing, p.ball_control as team_2p9_ballcontrol, p.reactions as team_2p9_reactions, p.balance as team_2p9_balance,
p.shot_power as team_2p9_shotpower, p.jumping as team_2p9_jumping, p.stamina as team_2p9_stamina, p.strength as team_2p9_strength, p.aggression as team_2p9_aggression, p.interceptions as team_2p9_interceptions,
p.positioning as team_2p9_positioning, p.penalties as team_2p9_penalties, p.marking as team_2p9_marking, p.standing_tackle as team_2p9_standingtackle, 
p.sliding_tackle as team_2p9_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att10 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p10_weight, 
p.overall_rating as team_2p10_overall,p.potential as team_2p10_potential,p.crossing as team_2p10_crossing,p.heading_accuracy as team_2p10_heading,p.short_passing as team_2p10_shortpass,p.finishing as team_2p10_finishing, 
p.sprint_speed as team_2p10_sprintspeed, p.long_shots as team_2p10_longshots, p.agility as team_2p10_agility, p.vision as team_2p10_vision, p.volleys as team_2p10_volleys, p.dribbling as team_2p10_dribbling, p.curve as team_2p10_curve, 
p.free_kick_accuracy as team_2p10_freekickacc, p.long_passing as team_2p10_longpassing, p.ball_control as team_2p10_ballcontrol, p.reactions as team_2p10_reactions, p.balance as team_2p10_balance,
p.shot_power as team_2p10_shotpower, p.jumping as team_2p10_jumping, p.stamina as team_2p10_stamina, p.strength as team_2p10_strength, p.aggression as team_2p10_aggression, p.interceptions as team_2p10_interceptions,
p.positioning as team_2p10_positioning, p.penalties as team_2p10_penalties, p.marking as team_2p10_marking, p.standing_tackle as team_2p10_standingtackle, 
p.sliding_tackle as team_2p10_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}

team_2f_att11 <- function(player_api_id, con){
filter = "AND m.player_api_id is NOT NULL"
p_att <- tbl_df(dbGetQuery(con,paste("SELECT m.height as p2_height, m.weight as team_2p11_weight, 
p.overall_rating as team_2p11_overall,p.potential as team_2p11_potential,p.crossing as team_2p11_crossing,p.heading_accuracy as team_2p11_heading,p.short_passing as team_2p11_shortpass,p.finishing as team_2p11_finishing, 
p.sprint_speed as team_2p11_sprintspeed, p.long_shots as team_2p11_longshots, p.agility as team_2p11_agility, p.vision as team_2p11_vision, p.volleys as team_2p11_volleys, p.dribbling as team_2p11_dribbling, p.curve as team_2p11_curve, 
p.free_kick_accuracy as team_2p11_freekickacc, p.long_passing as team_2p11_longpassing, p.ball_control as team_2p11_ballcontrol, p.reactions as team_2p11_reactions, p.balance as team_2p11_balance,
p.shot_power as team_2p11_shotpower, p.jumping as team_2p11_jumping, p.stamina as team_2p11_stamina, p.strength as team_2p11_strength, p.aggression as team_2p11_aggression, p.interceptions as team_2p11_interceptions,
p.positioning as team_2p11_positioning, p.penalties as team_2p11_penalties, p.marking as team_2p11_marking, p.standing_tackle as team_2p11_standingtackle, 
p.sliding_tackle as team_2p11_slidingtackle FROM Player_Attributes p,
Player m WHERE p.player_api_id=",player_api_id," AND 
m.player_api_id=", player_api_id,filter)))
}





## English Premier League Lineups (TOP 6)
    #Manchester United Lineup
MU <- (get_Lineup("10260", con))
    #Arsenal Lineup
ARS <-(get_Lineup("9825", con))
    #Chelsea Lineup
CHS <-(get_Lineup("8455", con))
    #Manchester City Lineup
MC <-(get_Lineup("8456", con))
    #Fulham Lineup
FUL<-(get_Lineup("9879", con))
    #Tottenham Lineup
TOT <-(get_Lineup("8586", con))



###

## Liga BBVA
    #Barcelona Lineup
BARC <-(get_Lineup("8634", con))
    #Real Madrid Lineup
REAL<-(get_Lineup("8633", con))
    #Sevilla Lineup
SEV<-(get_Lineup("8302", con))
    #Atletico Madrid Lineup
ATL_M<-(get_Lineup("9906", con))
    #Villareal Lineup
VIL<-(get_Lineup("10205", con))
    #Atletico Bilbao Lineup
ATL_BILB<-(get_Lineup("8315", con))

###

##Serie A
    #Juventus Lineup
JUV<-(get_Lineup("9885", con))##
    #Milan Lineup
MIL<-(get_Lineup("8564", con))##
    #Roma Lineup
ROM<-(get_Lineup("8686", con))
    #Inter Lineup
INT<-(get_Lineup("8636", con))
    #Lazio Lineup
LAZ<-(get_Lineup("8543", con))
    #Napoli Lineup
NAP<-(get_Lineup("9875", con))

###

##Bundesliga
    #Bayern Lineup
BFC<-(get_Lineup("9823", con))
    #Borussia Dortmund Lineup
DORT<-(get_Lineup("9789", con))
    #FC Schalke Lineup
SHLK<-(get_Lineup("10189", con)) ##
    #Wolfsburg Lineup
WOLF<-(get_Lineup("8721", con))
    #Hannover Lineup
HAN<-(get_Lineup("9904", con))
    #Bayern Leverkusen Lineup
BYL<-(get_Lineup("8178", con))

###




### Premier League Lineup Averages

MU_Lineup_Averages = (lineupAtt(MU,con))

ARS_Lineup_Averages = (lineupAtt(ARS,con))

CHS_Lineup_Averages = (lineupAtt(CHS,con))

MC_Lineup_Averages = (lineupAtt(MC,con))

FUL_Lineup_Averages = (lineupAtt(FUL,con))

TOT_Lineup_Averages = (lineupAtt(TOT,con))

### Liga BBVA Lineup Averages 
BARC_Lineup_Averages = (lineupAtt(BARC,con))

REAL_Lineup_Averages = (lineupAtt(REAL,con))

SEV_Lineup_Averages = (lineupAtt(SEV,con))

ATLM_Lineup_Averages = (lineupAtt(ATL_M,con))

VIL_Lineup_Averages = (lineupAtt(VIL,con))

ATLBILB_Lineup_Averages = (lineupAtt(ATL_BILB,con))

### Bundesliga




MU_Lineup_Averages




###team1 lineupAtt

team_1lineupAtt <- function(team_acronym, con){
 
att_mean = c()
i = 1
for (id in team_acronym$reference_number1){
    p_att = (team_1goalie_att(id, con))
    attributes = colMeans(p_att[i,3:10])
    i = i + 1
p1_att_mean_df = t(as.data.frame(attributes))
}


#p2
att_mean = c()
i=1
for (id in team_acronym$reference_number2){
    p_att = (team_1f_att2(id, con))
    p2 = colMeans(p_att[i,1:31])
    i = i + 1
p2_att_mean_df = t(as.data.frame(p2))
}



#p3
att_mean = c()
i=1
for (id in team_acronym$reference_number3){
    p_att = (team_1f_att3(id, con))
    p3 = colMeans(p_att[i,1:31])
    i = i + 1
p3_att_mean_df = t(as.data.frame(p3))
}



#p4
att_mean = c()
i=1
for (id in team_acronym$reference_number4){
    p_att = (team_1f_att4(id, con))
    p4 = colMeans(p_att[i,1:31])
    i = i + 1
p4_att_mean_df = t(as.data.frame(p4))
}

#p5
att_mean = c()
i=1
for (id in team_acronym$reference_number5){
    p_att = (team_1f_att5(id, con))
    p5 = colMeans(p_att[i,1:31])
    i = i + 1
p5_att_mean_df = t(as.data.frame(p5))
}
p5_att_mean_df 


#p6
att_mean = c()
i=1
for (id in team_acronym$reference_number6){
    p_att = (team_1f_att6(id, con))
    p6 = colMeans(p_att[i,1:31])
    i = i + 1
p6_att_mean_df = t(as.data.frame(p6))
}
 


#p7
att_mean = c()
i=1
for (id in team_acronym$reference_number7){
    p_att = (team_1f_att7(id, con))
    p7 = colMeans(p_att[i,1:31])
    i = i + 1
p7_att_mean_df = t(as.data.frame(p7))
}


#p8
att_mean = c()
i=1
for (id in team_acronym$reference_number8){
    p_att = (team_1f_att8(id, con))
    p8 = colMeans(p_att[i,1:31])
    i = i + 1
p8_att_mean_df = t(as.data.frame(p8))
}

#p9
att_mean = c()
i=1
for (id in team_acronym$reference_number9){
    p_att = (team_1f_att9(id, con))
    p9 = colMeans(p_att[i,1:31])
    i = i + 1
p9_att_mean_df = t(as.data.frame(p9))
}


#p10
att_mean = c()
i=1
for (id in team_acronym$reference_number10){
    p_att = (team_1f_att10(id, con))
    p10 = colMeans(p_att[i,1:31])
    i = i + 1
p10_att_mean_df = t(as.data.frame(p10))
}


#p11
att_mean = c()
i=1
for (id in team_acronym$reference_number11){
    p_att = (team_1f_att11(id, con))
    p11 = colMeans(p_att[i,1:31])
    i = i + 1
p11_att_mean_df = t(as.data.frame(p11))
}
Lineup_att_means = rbind(p2_att_mean_df,p3_att_mean_df,p4_att_mean_df,p5_att_mean_df,p6_att_mean_df,p7_att_mean_df,p8_att_mean_df,p9_att_mean_df,p10_att_mean_df,p11_att_mean_df)   

}

x = team_1lineupAtt(MC, con)
x




MC_atts  = team_1lineupAtt(MC, con)

TOT_atts = team_1lineupAtt(TOT, con)
FUL_atts = team_1lineupAtt(FUL, con)
ARS_atts = team_1lineupAtt(ARS, con)
MU_atts  = team_1lineupAtt(MU, con)
CHS_atts = team_1lineupAtt(CHS, con)




test = as.data.frame(MC_atts)
test$team = "MC"
test$player = rownames(test)
MC_atts = test


test = as.data.frame(TOT_atts)
test$team = "TOT"
test$player = rownames(test)
TOT_atts = test


test = as.data.frame(FUL_atts)
test$team = "FUL"
test$player = rownames(test)
FUL_atts = test


test = as.data.frame(ARS_atts)
test$team = "ARS"
test$player = rownames(test)
ARS_atts = test


test = as.data.frame(CHS_atts)
test$team = "CHS"
test$player = rownames(test)
CHS_atts = test

test = as.data.frame(MU_atts)
test$team = "MU"
test$player = rownames(test)
MU_atts = test

Premier_atts = rbind(MC_atts, ARS_atts, CHS_atts, FUL_atts, MU_atts, TOT_atts)
Premier_atts









ggplot(data=Premier_atts, aes(x=player, y=team_1p2_overall, fill=team)) + 
theme_classic() +
geom_bar(stat="identity", width = .5, position=position_dodge(width= .3)) + 
scale_fill_hue(name="Team") +
xlab("Position") +
ylab("Overall Rating")




###team2 lineupAtt

team_2lineupAtt <- function(team_acronym, con){
 
att_mean = c()
i = 1
for (id in team_acronym$reference_number1){
    p_att = (team_2goalie_att(id, con))
    attributes = colMeans(p_att[i,3:10])
    i = i + 1
p1_att_mean_df = t(as.data.frame(attributes))
}


#p2
att_mean = c()
i=1
for (id in team_acronym$reference_number2){
    p_att = (team_2f_att2(id, con))
    attributes2 = colMeans(p_att[i,1:31])
    i = i + 1
p2_att_mean_df = t(as.data.frame(attributes2))
}



#p3
att_mean = c()
i=1
for (id in team_acronym$reference_number3){
    p_att = (team_2f_att3(id, con))
    p3 = colMeans(p_att[i,1:31])
    i = i + 1
p3_att_mean_df = t(as.data.frame(p3))
}



#p4
att_mean = c()
i=1
for (id in team_acronym$reference_number4){
    p_att = (team_2f_att4(id, con))
    p4 = colMeans(p_att[i,1:31])
    i = i + 1
p4_att_mean_df = t(as.data.frame(p4))
}

#p5
att_mean = c()
i=1
for (id in team_acronym$reference_number5){
    p_att = (team_2f_att5(id, con))
    p5 = colMeans(p_att[i,1:31])
    i = i + 1
p5_att_mean_df = t(as.data.frame(p5))
}
p5_att_mean_df 


#p6
att_mean = c()
i=1
for (id in team_acronym$reference_number6){
    p_att = (team_2f_att6(id, con))
    p6 = colMeans(p_att[i,1:31])
    i = i + 1
p6_att_mean_df = t(as.data.frame(p6))
}
 


#p7
att_mean = c()
i=1
for (id in team_acronym$reference_number7){
    p_att = (team_2f_att7(id, con))
    p7 = colMeans(p_att[i,1:31])
    i = i + 1
p7_att_mean_df = t(as.data.frame(p7))
}


#p8
att_mean = c()
i=1
for (id in team_acronym$reference_number8){
    p_att = (team_2f_att8(id, con))
    p8 = colMeans(p_att[i,1:31])
    i = i + 1
p8_att_mean_df = t(as.data.frame(p8))
}

#p9
att_mean = c()
i=1
for (id in team_acronym$reference_number9){
    p_att = (team_2f_att9(id, con))
    p9 = colMeans(p_att[i,1:31])
    i = i + 1
p9_att_mean_df = t(as.data.frame(p9))
}


#p10
att_mean = c()
i=1
for (id in team_acronym$reference_number10){
    p_att = (team_2f_att10(id, con))
    p10 = colMeans(p_att[i,1:31])
    i = i + 1
p10_att_mean_df = t(as.data.frame(p10))
}


#p11
att_mean = c()
i=1
for (id in team_acronym$reference_number11){
    p_att = (team_2f_att11(id, con))
    p11 = colMeans(p_att[i,1:31])
    i = i + 1
p11_att_mean_df = t(as.data.frame(p11))
}
Lineup_att_means = cbind(p1_att_mean_df,p2_att_mean_df,p3_att_mean_df,p4_att_mean_df,p5_att_mean_df,p6_att_mean_df,p7_att_mean_df,p8_att_mean_df,p9_att_mean_df,p10_att_mean_df,p11_att_mean_df)   

}

x = team_2lineupAtt(MC, con)
x
ncol(x)
###




matchUp <- function(team1_api_id, team2_api_id, con){
    
    team_1 <- get_Lineup(team1_api_id, con)
    team_2 <- get_Lineup(team2_api_id, con)
    
    team_1_att <- team_1lineupAtt(team_1, con)
    team_2_att <- team_2lineupAtt(team_2, con)
    
    
    team1_is_home = tbl_df(dbGetQuery(con,paste("SELECT home_team_goal as team_1_goal, away_team_goal as team_2_goal, home_team_api_id, away_team_api_id FROM Match WHERE home_team_api_id =", team1_api_id,"AND away_team_api_id =", team2_api_id)))
    team2_is_home = tbl_df(dbGetQuery(con,paste("SELECT home_team_goal as team_2_goal, away_team_goal as team_1_goal, home_team_api_id, away_team_api_id FROM Match WHERE home_team_api_id =", team2_api_id,"AND away_team_api_id =", team1_api_id)))
    
    matchups = rbind(team1_is_home, team2_is_home)
    
    team_1won = as.numeric(matchups$team_1_goal > matchups$team_2_goal)
    team_2won = as.numeric(matchups$team_2_goal > matchups$team_1_goal)
    tie       = as.numeric(matchups$team_2_goal == matchups$team_1_goal)
    
    matchups = cbind(team_1won, team_2won, tie, matchups)
    
    

    matrix <- cbind(matchups, team_1_att, team_2_att)
    matrix 
    
    #output = c("Team 1 won ", team1_wins," times. Team 2 won ", team2_wins," times. Team 1 tied at home ", team1_home_ties, " times. Team 2 tied at home ", team2_home_ties," times.")
    #print(output)
    
    
    
}
t = matchUp(9825,8455,con)    
t




install.packages('ggbiplot')
library(ggbiplot)
home_wins = select.match$home_wins
g <- ggbiplot(matrix.pca, obs.scale = 1, var.scale = 1, 
              groups = team2_wins, ellipse = TRUE, 
              circle = TRUE)
g <- g + scale_color_discrete(name = '')
g <- g + theme(legend.direction = 'horizontal', 
               legend.position = 'top')
print(g)





str(MU_Lineup_Averages)
rownames(MU_Lineup_Averages)
colnames(MU_Lineup_Averages)

heatmap(as.matrix(MU_Lineup_Averages)
        , scale = "column"
        , col=heat.colors(256)
        , main = "Attributes of Manchester United by position"
        ,Colv=NA
        )




summary(MU_Lineup_Averages)
get_ipython().run_line_magic('pinfo', 'barplot')

or <- MU_Lineup_Averages[2:10,1:3]
or.d <- density(or)
or.d
barplot(or)




"""
player  <- select(player,player_api_id, player_name) # use player_api_id as key for join
Team    <- select(Team, team_api_id, team_long_name, team_short_name) # use team_api_id as key for join
Country <-select(Country, id, name) %>% rename(country_id = id)  %>% rename(country_name = name)   # use country_id as key for join
League  <- select(League, country_id, name) %>% rename(league_name = name) # use country_id as key for join
Match   <-select(Match, id, country_id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11, goal, shoton, shotoff, foulcommit, card, cross, corner, possession)
"""




MU_Lineup_Attributes
melted <- colMeans(MU_Lineup_Attributes)
melted

