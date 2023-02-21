#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




#Checking data
data = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()




data.describe()




#Changing attrition into boolean
data["Attrition"] = np.where(data['Attrition']=='Yes', 1, 0)




data.head()




#Checking for null values
data.isnull().sum()




sns.countplot('Attrition', data=data)




count_attrition = pd.DataFrame(data['Attrition'].value_counts())
count_attrition['percentage'] = round((count_attrition["Attrition"]/data.Attrition.count())*100,2)
count_attrition




# we can see that only 16% seem to fall into attrition, it's very unbalananced to create a model based on this data




count_attrition




#Checking distribution
plt.hist(data["Age"], bins=20)




#We can see that the Age is sort of normally distributed and that the mean is around 37 years old
data["Age"].mean()




#We proceed to make a pitvot table to explore the different variables based on attrition (Yes and No)
t=data.pivot_table(index='Attrition', aggfunc='mean')
t




t.Age




t.MonthlyIncome




t.JobSatisfaction




#Then we decided to explore the variables we considered important for a predictive model such as Age, Monthly Income and Job Satisfaction
#The results show that the age average for attrition is lower which shows that older employees tend to keep their jobs
#Then the montly income shows the obvious, the employees with lower salaries tend to leave the company.
#There is a slightly difference in Job Satisfaction where employees with lower satisfaction tend to leave.




x=data[data.Attrition==0]
y = data[data.Attrition==1]




plt.figure(figsize=(10,5))

plt.subplot(111)

sns.distplot(x.Age, color='red')
sns.distplot(y.Age,  color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Age')




#We also wanted to review the age variable more into depth and the distributions shows that the younger people tend to 
#quit their job more often. The mean age for attrition is 33 years old. It just proved what we observed in the previous pivot table




#Checking the distributions
plt.figure(figsize=(10,5))

plt.subplot(221)

sns.kdeplot(x.MonthlyIncome, shade= True, color='red')
sns.kdeplot(y.MonthlyIncome, shade= True, color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Monthly Income')

plt.subplot(222)
sns.kdeplot(x.DailyRate, shade= True, color='red')
sns.kdeplot(y.DailyRate, shade= True, color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Daily')

plt.subplot(223)
sns.kdeplot(x.MonthlyRate, shade= True, color='red')
sns.kdeplot(y.MonthlyRate, shade= True, color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Monthly Rate')

plt.subplot(224)
sns.kdeplot(x.HourlyRate, shade= True, color='red')
sns.kdeplot(y.HourlyRate, shade= True, color='blue')
plt.legend(title="Attrition", loc='upper right', labels = ["No", "Yes"])
plt.title('Hourly Rate')
plt.tight_layout()

plt.show()




#The monthly income distribution seem to be supporting our theory that the people that are earning the lest are more likely to leave
# Also the people who are earning less on a daily basis seem to be more likely to leave
#It can be also noticeable in the Hourly rate in a lower proportion but still follows the logic
#However, it is curious to review the Montly Rate distribution because it does not follow the logic




get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1575998527239' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard2_15759984165290&#47;Dashboard2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Dashboard2_15759984165290&#47;Dashboard2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard2_15759984165290&#47;Dashboard2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1575998527239');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")




get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1575998582864' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard1_15759969896590&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Dashboard1_15759969896590&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard1_15759969896590&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1575998582864');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")




pd.set_option('display.max_columns', 30)




thresh = 0.4




correlation = data.corr()




correlation

