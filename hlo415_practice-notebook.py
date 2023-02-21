#!/usr/bin/env python
# coding: utf-8



library(data.table)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)




ggplot(train[,.N,by=weekday_alta],aes(x = weekday_alta,y = N,fill=weekday_alta))+
  geom_bar(stat="identity")+ggtitle("Number of customers that became 'first holder' by day of week")




ggplot(train[year_alta>2009,.N,by=.(month_alta,year_alta)],aes(x = month_alta,y=N,fill=month_alta))+
  geom_bar(stat="identity")+ggtitle("Number of customers that became 'first holder' by month and year")+
  facet_wrap(~year_alta)

