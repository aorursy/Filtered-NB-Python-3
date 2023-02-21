#!/usr/bin/env python
# coding: utf-8



library("dplyr")
library(ggplot2)

system("ls ../input")
zika = read.csv("../input/cdc_zika.csv", stringsAsFactors = F, header = T)

# cleaning up empty values and cases = 'municipalities'
zika1 <- subset(zika, !is.na(value) & unit == 'cases')

# treatment for country
my_country_name <- c("ARGENTINA","BRAZIL","COLOMBIA","DOMINICAN REPUBLIC","ECUADOR","EL SALVADOR","GUATEMALA","HAITI","MEXICO","NICARAGUA","PANAMA","PUERTO RICO","UNITED STATES","VIRGIN ISLANDS, U.S.")
names(my_country_name) <- c("AR","BR", "CO", "DO", "EC", "SV", "GT", "HT", "MX", "NI", "PA", "PR", "US", "VI")

# fixing code for Haiti
zika1$country_code <- ifelse(substr(zika1$data_field_code,1, 2) == "HA", "HT", substr(zika1$data_field_code,1, 2)) 
zika1$country <- as.character(my_country_name[zika1$country_code])

# year_month - new field
zika1$year_month <- format(as.Date(zika1$report_date,format="%Y-%m-%d"), "%Y-%m")

# year - new field
zika1$year <- format(as.Date(zika1$report_date,format="%Y-%m-%d"), "%Y")

# Treatment to remove redundat records
my_location_code <- c(10,9,8,7,6,5,4,3,2,1)
names(my_location_code) <- c("country","territory","region","province","state","county","municipality","department","district","city")
zika1$location_weight <- as.character(my_location_code[zika1$location_type])

zika2 <- summarise(group_by(zika1, report_date, data_field_code),
                   min_location = min(location_weight))

zika1$identifier <- paste(zika1$report_date, zika1$data_field_code, sep="/")
zika2$identifier <- paste(zika2$report_date, zika2$data_field_code, sep="/")

# attempt to create a dictionary
min_locations <- zika2$min_location
names(min_locations) <- zika2$identifier
zika1$min_location <- as.character(min_locations[zika1$identifier])

zika1$value <- strtoi(zika1$value)

# removing reduntant records to the same report_date + data_field_code
my_data <- filter(zika1, !is.na(value) & location_weight == min_location )





# Zika cases evolution by country
ggplot(data=my_data,
       aes(x=year_month,
           y=value,
           fill=country))+
  geom_bar(stat='identity') + 
  xlab('Year-Month') + 
  ylab('Number of cases') +
  ggtitle('Number of cases by time')


# Evolution in the top 6 countries
my_data_by_country <- head(arrange(summarise(group_by(my_data, country),cases=sum(value)),desc(cases)),n=6)
top6 <- my_data_by_country$country
ggplot(data=subset(my_data, country %in% top6),
       aes(x=year_month,
           y=value,
           fill=country))+
  geom_bar(stat='identity') + 
  xlab('Country') + 
  ylab('Number of cases') +
  ggtitle('Number of cases by time - Top 6 Countries')

# Brazil shows a huge decrease in the last months. Based on recent news, I believe it is missing data.

# Number of cases for 6 top countries

ggplot(my_data_by_country, 
       aes(x = reorder(country, desc(cases)), 
           y = cases,
           fill = cases)) + 
  geom_bar(stat='identity') + 
  xlab('Country') + 
  ylab('Number of cases') +
  ggtitle('Number of cases - Top 6 Countries')

# Brazil and Colombia are the worst countries and it justifies a more detailed view.

my_data_top2 <- head(arrange(summarise(group_by(my_data, country),cases=sum(value)),desc(cases)),n=2)
top2 <- my_data_top2$country
ggplot(data=subset(my_data, country %in% top2),
       aes(x = reorder(country, desc(value)), 
           y = value,
           fill = data_field)) + 
  geom_bar(stat='identity') + 
  xlab('Country') + 
  ylab('Number of cases') +
  ggtitle('Cases - Top 2 countries')

# For both countries, I'm wondering if "zika_suspected" and "zika-reported" will turn to zika_confirmed. If it is the case, many records may be redundants.

# Microcephaly cases

microcephaly <- c("microcephaly_confirmed","microcephaly_confirmed_cumulative","microcephaly_fatal_confirmed","microcephaly_fatal_not","microcephaly_fatal_under_investigation","microcephaly_not","microcephaly_suspected","microcephaly_suspected_4weeks","microcephaly_suspected_cumulative","microcephaly_under_investigation")
ggplot(data=subset(my_data, data_field %in% microcephaly),
       aes(x = reorder(country, desc(value)), 
           y = value,
           fill = data_field)) + 
  geom_bar(stat='identity') + 
  xlab('Country') + 
  ylab('Number of cases') +
  ggtitle('Microcephaly cases')

