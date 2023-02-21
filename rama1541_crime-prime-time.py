#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from __future__ import division




df = pd.read_csv("../input/crime.csv")




# change column name "Text_General_Code"
# transform column names to lowercase for easier handling with pandas

column_names = []
for column in df.columns:
    if column == "Text_General_Code":
        column_names.append("crime")
    else:
        column_names.append(column.lower())

df.columns = column_names




def wrangle_year(date):
    return int(date.split("-")[0])

df["year"] = df.dispatch_date.apply(wrangle_year)

# excluding 2017 since there is not much data yet
df = df[df.year!=2017]




def wrangle_month(date):
    return int(date.split("-")[1])

df["month"] = df.month.apply(wrangle_month)




def wrangle_day(date):
    return int(date.split("-")[2])

df["day"] = df.dispatch_date.apply(wrangle_day)




# total number of crimes for each month
crime_monthly = df.month.value_counts().sort_index()

# calculating the average number of crimes per day for the given month
month_total_days = df.groupby(["year", "month", "day"]).size().unstack("month").count()
crime_monthly_per_day = crime_monthly / month_total_days

# main plot
crime_monthly_per_day.plot(marker="o", xticks=range(1, 13), title="Crime Prime Time during the Year", color="#018571")                      .set(xlabel="Month", ylabel="Average Number of Crimes per Day")

# plotting additional information
average_per_day = df.dispatch_date.value_counts().mean()    
plt.hlines(average_per_day, 1, 12, alpha=0.8, linestyle="--", color="#8c510a")
plt.text(1.1, average_per_day+1, "overall average", color="#8c510a")




# creating the column "day_of_the_week"

# the 1st of January 2006 was a Sunday
day_of_the_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
dates = sorted(df.dispatch_date.unique())
number_weeks = int(len(dates) / 7)

df_dotw = pd.DataFrame({"date": dates, "day_of_the_week": day_of_the_week * number_weeks})
df = pd.merge(df, df_dotw, left_on="dispatch_date", right_on="date")




# total number of crimes for each day of the week
crime_daily = df.day_of_the_week.value_counts()

# putting the days in the right order
crime_daily = crime_daily[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]

# calculating average
total_days = len(df.dispatch_date.unique())
crime_daily_average = crime_daily / (total_days / 7)

# plotting
crime_daily_average.plot(marker="o", title="Crime Prime Time during the Week", color="#018571")                    .set(xlabel="Day of the Week",  ylabel="Average Number of Crimes")
    
# plotting additional information
average_per_day = df.dispatch_date.value_counts().mean()    
plt.hlines(average_per_day, 0, 6, alpha=0.8, linestyle="--", color="#8c510a")
plt.text(1.1, average_per_day+1, "overall average", color="#8c510a")




# average number of crimes for each hour of the day
crime_hourly = df.hour.value_counts().sort_index() / total_days

# main plot
crime_hourly.plot(xticks=range(24), figsize=(17,5), title="Crime Prime Time during the Day", color="#018571")             .set(xlabel="Hour",  ylabel="Average Number of Crimes")




crime_hourly_dofw = pd.crosstab(df.hour, df.day_of_the_week) / (total_days/7)
crime_hourly_dofw = crime_hourly_dofw[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]

crime_hourly_dofw.plot(xticks=range(24), figsize=(17,5), title="Crime Prime Time during the Day", cmap="Accent")                  .set(xlabel="Hour",  ylabel="Average Number of Crimes")




# combining the different homicide types into one
def combining_homicide(crime):
    if pd.isnull(crime):
        return np.nan
    if "Homicide" in crime:
        return "Homicide - Criminal"
    else:
        return crime

df["crime"] = df.crime.apply(combining_homicide)




crime_types = pd.crosstab(df.hour, df.crime)

# dropping non-specific categories
crime_types = crime_types.drop(["All Other Offenses", "Other Assaults"], axis=1)

# using relative numbers to enable comparisons between the different types
# since some types are much more numerous than outhers
crime_types = crime_types.div(crime_types.sum())




# I manually allocated the crime types into these groups
bio_rythm = [1,4,6,13,15,19,20,21,23,27,28]
nine_to_five = [7,8,9,16,25]
night_shift = [0,2,5,11,12,14,17,18,22,26]
other = [3,10,24]




fig, ax = plt.subplots(2,2, figsize=(15,12), sharey=True)

plot_1 = crime_types[bio_rythm].plot(ax=ax[0][0], cmap="Accent") 
plot_1.set(ylabel="Percent", xlabel="Hour")
plot_1.set_title("Biological Rhythm", fontweight="bold")

plot_2 = crime_types[nine_to_five].plot(ax=ax[0][1], cmap="Accent") 
plot_2.set(ylabel="Percent", xlabel="Hour")
plot_2.set_title("9-to-5", fontweight="bold")

plot_3 = crime_types[night_shift].plot(ax=ax[1][0], cmap="Accent")
plot_3.set(ylabel="Percent", xlabel="Hour")
plot_3.set_title("Night Shift", fontweight="bold")

plot_4 = crime_types[other].plot(ax=ax[1][1], cmap="Accent")
plot_4.set(ylabel="Percent", xlabel="Hour")
plot_4.set_title("Other", fontweight="bold")




# calculating the average number of crimes for the respective day of the year
years_of_data = 11
holidays = df.groupby(["month", "day"]).size() / years_of_data

# the 29th of February occured only 3-times during the period that the data set covers 
holidays[2][29] = (holidays[2][29] * years_of_data) / 3




# making sure that the xticks in the following graph are always on the first of the month
days_of_month_leap_year = [31,29,31,30,31,30,31,31,30,31,30,31]
ticks_leap_year = []

n = 0
for days in days_of_month_leap_year:
    ticks_leap_year.append(n)
    n += days




plot = holidays.plot(figsize=(15,6), xticks=ticks_leap_year, color="#018571")
plot.set(xlabel="(Month, Day)", ylabel="Average Number of Crimes")
plot.set_title("Do criminals celebrate holidays?", fontweight="bold", fontsize=20)
    
# Independence Day
plt.arrow(152,435,20,20, width=0.5, color="k", head_starts_at_zero=False)
plt.text(128,420, "Independence Day")

# Thanksgiving
plt.arrow(295,420,20,20, width=0.5, color="k", head_starts_at_zero=False)
plt.text(275,402, "Thanksgiving")

# Christmas
plt.arrow(320,220,20,0, width=0.5, color="k", head_starts_at_zero=False)
plt.text(295,218, "Christmas")






