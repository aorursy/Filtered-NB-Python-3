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




---
title: "Top 20 football players"
author: "Eryk Walczak"
date: "13 July 2016"
output: 
  html_document:
    toc: true
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Loading data

Data is stored in SQLite file so first I have to connect to the data base. 
Then I look at the available tables and choose the ones I'm interested in. 

*player* and *player_stats* will be joined because they contain key information.

```{r, message=FALSE, warning=FALSE}
library(dplyr)
library(RSQLite)

con <- dbConnect(SQLite(), dbname="../input/database.sqlite")

# list all tables
dbListTables(con)

player       <- tbl_df(dbGetQuery(con,"SELECT * FROM player"))
player_stats <- tbl_df(dbGetQuery(con,"SELECT * FROM player_stats"))

player_stats <-  player_stats %>%
  rename(player_stats_id = id) %>%
  left_join(player, by = "player_api_id")
```

## Browsing data

I want to see what's available in my joined table:

```{r}
str(player_stats)
```

There are several observations in *date_stat* so I choose the latest:

```{r}
latest_ps <- 
  player_stats %>% 
  group_by(player_api_id) %>% 
  top_n(n = 1, wt = date_stat) %>%
  as.data.frame()
```

I'm only interested in top 20 players so I choose them from the latest observation based on *overall_rating*:

```{r}
top20 <- 
  latest_ps %>% 
  arrange(desc(overall_rating)) %>% 
  head(n = 20) %>%
  as.data.frame()
```

Here are the key fields after filtering:

```{r, message=FALSE, warning=FALSE}
library(DT)

top20 %>% 
  select(player_name, birthday, height, weight, preferred_foot, overall_rating) %>% 
  datatable(., options = list(pageLength = 10))
```

## Charts

### Overall scores

I will start with describing the distribution of overall scores

```{r, message=FALSE, warning=FALSE}
library(DescTools)

Desc(top20$overall_rating, plotit = TRUE)
```

The scores are high as expected from the top football players

### Correlation matrix

In order to see correlations between the variables I decided to create an interactive correlation matrix.
*click on data point to see a scatter plot*

```{r, message=FALSE, warning=FALSE}
library(qtlcharts)

iplotCorr(top20[,10:42], reorder=TRUE)
```

### Scatter plots

I wanted to see what's the relationship between *overall_score* and other numeric variables so I made an interactive scatter plot:

```{r, message=FALSE, warning=FALSE}
library(ggvis)

measures <- names(top20[,10:42])

top20 %>% 
  ggvis(x = input_select(measures, label = "Choose the x-axis:", map = as.name)) %>% 
  layer_points(y = ~overall_rating, fill = ~player_name)

```

### Radar chart

To see a complete profile of individual players I create a radar chart. This is a great way to plot multiple variables.
Data first had to be transformed to work with the [radarchart](https://github.com/MangoTheCat/radarchart/blob/master/vignettes/preparingData.Rmd) package. 

```{r, message=FALSE, warning=FALSE}
library(radarchart)
library(tidyr)

radarDF <- top20 %>% select(player_name, 10:42) %>% as.data.frame()

radarDF <- gather(radarDF, key=Label, value=Score, -player_name) %>%
  spread(key=player_name, value=Score)

chartJSRadar(scores = radarDF, maxScale = 100, showToolTipLabel = TRUE)
```

## Summary

This should be a good start to play with the players' data.
More analyses to follow...

