#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re  # Regex Operations
pd.options.mode.chained_assignment = None

# Visualizations
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns


from tabulate import tabulate

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




df=pd.read_csv('../input/loan.csv',low_memory=False)




df.head()




df.columns.tolist()




msno.bar(df)




msno.heatmap(df)




msno.dendrogram(df)




loan_status_distribution=pd.DataFrame(df['loan_status'].value_counts())
loan_status_distribution.reset_index(inplace=True)
loan_status_distribution.columns=['Loan_Status_Category','Number of applicants']
loan_status_distribution




loan_status_distribution['Percentage of Applicants']=loan_status_distribution['Number of applicants'].apply(lambda x: float(x)/len(df))
loan_status_distribution[['Loan_Status_Category','Percentage of Applicants']] 




trace=go.Pie(labels=loan_status_distribution['Loan_Status_Category'],values=loan_status_distribution['Number of applicants'])
layout=go.Layout(showlegend=False)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig,filename='pie')




from wordcloud import WordCloud, STOPWORDS
from PIL import Image

df['new_title']=df['title'].apply(lambda x: str(x).lower().split())
a=[j for i in df['new_title'] for j in i]
text=','.join(a)

stopwords = set(STOPWORDS)
wc = WordCloud(background_color="white", max_words=50,stopwords=stopwords,min_font_size=2,collocations=False,prefer_horizontal=1               ,relative_scaling=.3,colormap='plasma')
wc.generate(text)

# show
plt.figure(figsize=(12,15))
plt.imshow(wc,interpolation='bilinear')
plt.axis("off")
plt.show()




df['loan_status']=df['loan_status'].apply(lambda x: 'Fully Paid' if 'Fully Paid' in str(x) else x)
df['loan_status']=df['loan_status'].apply(lambda x: 'Charged Off' if 'Charged Off' in str(x) else x)
df['loan_status']=df['loan_status'].apply(lambda x: 'Late' if ('Late' in str(x) or 'Grace' in str(x)) else x)

sns.set_style('whitegrid')
plt.figure(figsize=(10,10))
sns.countplot(x='loan_status',data=df,palette='cubehelix')
sns.despine(top=True,right=True)
plt.xticks(rotation=90)
plt.show()




loan_status_distribution=pd.DataFrame(df['loan_status'].value_counts())
loan_status_distribution.reset_index(inplace=True)
loan_status_distribution.columns=['Loan_Status_Category','Number of applicants']
loan_status_distribution['Percentage of Applicants']=loan_status_distribution['Number of applicants'].apply(lambda x: float(x)/len(df))
loan_status_distribution[['Loan_Status_Category','Percentage of Applicants']]         

trace=go.Pie(labels=loan_status_distribution['Loan_Status_Category'],values=loan_status_distribution['Number of applicants'])
iplot([trace]) 




# First we will strip left and right trailing spaces
df['emp_title']=df['emp_title'].apply(lambda x: str(x).strip())

#Lower the title so that we can compare and combine employee job title
df['emp_title']=df['emp_title'].apply(lambda x: str(x).lower())

#Regex operatiosn to check for various employee job title
df['emp_title']=df['emp_title'].str.replace(r'\w+\smanager$','manager')
df['emp_title']= df['emp_title'].apply(lambda x: 'registered nurse' if (x=='rn' or x=='nurse') else x) #RN means Registered nurse
df['emp_title']=df['emp_title'].str.replace(r'\w+\steacher$','teacher')
df['emp_title']=df['emp_title'].str.replace(r'teacher\w+$','teacher')
df['emp_title']=df['emp_title'].str.replace(r'\w+\sdriver$','driver')
df['emp_title']=df['emp_title'].str.replace(r'\w+\sengineer$','engineer')


df['emp_title']=df['emp_title'].apply(lambda x: 'teacher' if ('teacher') in str(x) else x)
df['emp_title']=df['emp_title'].apply(lambda x: 'manager' if ('manager') in str(x) else x)
df['emp_title']=df['emp_title'].apply(lambda x: 'nurse' if ('nurse') in str(x) else x)
df['emp_title']=df['emp_title'].apply(lambda x: 'driver' if ('driver') in str(x) else x)
df['emp_title']=df['emp_title'].apply(lambda x: 'engineer' if ('engineer') in str(x) else x)
df['emp_title']=df['emp_title'].apply(lambda x: 'analyst' if ('analyst') in str(x) else x)
df['emp_title']=df['emp_title'].apply(lambda x: 'accountant' if ('accountant') in str(x) else x)
df['emp_title']=df['emp_title'].apply(lambda x: 'assistant' if ('assistant') in str(x) else x)
df['emp_title']=df['emp_title'].apply(lambda x: 'director' if ('director') in str(x) else x)

#Capitalize first letter of each word in the employee title
df['emp_title'] = df['emp_title'].apply(lambda x: str(x).title()) # Employee title contains 

emp_title=df['emp_title'].value_counts()[:10].index
df['emp_title']=df['emp_title'].apply(lambda x: x if str(x) in emp_title  else 'OTHER')
df['emp_title']=df['emp_title'].apply(lambda x: 'OTHER' if str(x) in 'Nan' else x)
df['emp_title'].value_counts()




#create a dataframe that contains the count of top 10 job titles
a=pd.DataFrame(df.emp_title.value_counts()) # 1 omitted as it contains missing data
a.reset_index(inplace=True)

#Color scheme
colors = ['rgb(165,0,38)', 'rgb(215,48,39)', 'rgb(244,109,67)',
         'rgb(253,174,97)','rgb(254,224,144)','rgb(224,243,248)',
         'rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)','rgb(49,54,149)']


trace1 = go.Bar(
    x=a['index'],
    y=a.emp_title,
    name='Employee',
    marker=dict(color=colors)
)
layout=go.Layout(title='Applicant Count by Job Title',xaxis=dict(title='Job Title'),yaxis=dict(title='Count'))
annotations = []

for i in range(0, len(a)):
    annotations.append(dict(x=a['index'][i], y=a['emp_title'][i]+1500, text=a['emp_title'][i],
                                  font=dict(family='Arial', size=10,
                                  color='rgba(0,0,0,1)'),
                                  showarrow=True,))
    layout['annotations'] = annotations

data=[trace1]
fig = go.Figure(data=data,layout=layout)
iplot(fig, filename='grouped-bar')




df['home_ownership']=df['home_ownership'].apply(lambda x: 'OTHER' if (x=='NONE' or x=='ANY') else x)


#create a dataframe that contains the count of top 10 job titles
a=pd.DataFrame(df.home_ownership.value_counts()) # 1 omitted as it contains missing data
a.reset_index(inplace=True)

#Color scheme
colors = ['rgb(165,0,38)', 'rgb(215,48,39)', 'rgb(244,109,67)',
         'rgb(253,174,97)','rgb(254,224,144)','rgb(224,243,248)',
         'rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)','rgb(49,54,149)']


trace1 = go.Bar(
    x=a['index'],
    y=a.home_ownership,
    name='Employee',
    marker=dict(color=colors)
)
layout=go.Layout(title='Applicant Count by Home Ownership',xaxis=dict(title='Job Title'),yaxis=dict(title='Count'))
annotations = []

for i in range(0, len(a)):
    annotations.append(dict(x=a['index'][i], y=a['home_ownership'][i]+1500, text=a['home_ownership'][i],
                                  font=dict(family='Arial', size=10,
                                  color='rgba(0,0,0,1)'),
                                  showarrow=True,))
    layout['annotations'] = annotations

data=[trace1]
fig = dict(data=data)
iplot(fig)




plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.countplot(x='emp_length',data=df,hue='grade',hue_order=['A','B','C','D','E','F','G'],
              order=['< 1 year','1 year','2 years','3 years', '4 years','5 years','6 years',
                     '7 years','8 years', '9 years','10+ years'],palette='cubehelix')
plt.xlabel('Applicant Current Job Length',size=20)
plt.ylabel('Count',size=20)
plt.xticks(size=14,rotation=90)
plt.yticks(size=14)
plt.title('Applicant Current Job length by grade of loan',size=20,y=1.05)
plt.show()




trace=[0]*6
grades=sorted(df['grade'].unique())
for i in range(0,(df['grade'].nunique())-1):
    trace[i] = go.Box(x=df['loan_amnt'][df['grade']==grades[i]],name = 'Grade '+grades[i])
layout=go.Layout(title='Loan Amount Distribution by Grades',xaxis=dict(title='Loan Amount'),yaxis=dict(title='Grade of Loan'))
fig=go.Figure(data=trace,layout=layout)
iplot(fig)




a=pd.DataFrame(df.groupby(['purpose']).mean()['int_rate']).sort_values(by='int_rate',ascending=False).reset_index()

#Color scheme
colors = ['rgb(165,0,38)', 'rgb(215,48,39)', 'rgb(244,109,67)','rgb(253,174,97)','rgb(254,224,144)',
          'rgb(224,243,248)','rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)', 'rgb(49,54,149)',
          'rgb(165,0,38)', 'rgb(215,48,39)', 'rgb(244,109,67)','rgb(253,174,97)']


trace1 = go.Bar(
    x=a['purpose'],
    y=a.int_rate,
    name='Interest Rate',
    marker=dict(color=colors)
)
layout=go.Layout(title='Average Interest rates by Loan Puprose',
                 xaxis=dict(title='Loan Purpose',tickangle=30),
                 yaxis=dict(title='Average Interest Rate'))
annotations = []

for i in range(0, 14):
    annotations.append(dict(x=a.purpose[i], y=a.int_rate[i]+0.8, text="%.2F" % a.int_rate[i]+"%",
                                  font=dict(family='Arial', size=10,
                                  color='rgba(0,0,0,1)'),
                                  showarrow=True,))
    layout['annotations'] = annotations

data=[trace1]
fig = go.Figure(data=data,layout=layout)
iplot(fig, filename='grouped-bar')




df['purpose']=df['purpose'].apply(lambda x: 'other' if (x=='renewable_energy' or x=='small_business'
                                                        or x=='vacation' or x=='wedding'
                                                       or x=='major_purchase') else x)

df['purpose']=df['purpose'].apply(lambda x: 'house' if (x=='moving' or x=='home_improvement') else x)




a=pd.DataFrame(df.groupby(['purpose']).mean()['int_rate']).sort_values(by='int_rate',ascending=False).reset_index()

#Color scheme
colors = ['rgb(165,0,38)', 'rgb(215,48,39)', 'rgb(244,109,67)','rgb(253,174,97)','rgb(254,224,144)',
          'rgb(224,243,248)','rgb(171,217,233)','rgb(116,173,209)','rgb(69,117,180)', 'rgb(49,54,149)',
          'rgb(165,0,38)', 'rgb(215,48,39)', 'rgb(244,109,67)','rgb(253,174,97)']


trace1 = go.Bar(
    x=a['purpose'],
    y=a.int_rate,
    name='Interest Rate',
    marker=dict(color=colors)
)
layout=go.Layout(title='Average Interest rates by Regrouped Loan Puprose',
                 xaxis=dict(title='Loan Purpose',tickangle=30),
                 yaxis=dict(title='Average Interest Rate'))
annotations = []

for i in range(0, len(a)):
    annotations.append(dict(x=a.purpose[i], y=a.int_rate[i]+0.8, text="%.2F" % a.int_rate[i]+"%",
                                  font=dict(family='Arial', size=10,
                                  color='rgba(0,0,0,1)'),
                                  showarrow=True,))
    layout['annotations'] = annotations

data=[trace1]
fig = go.Figure(data=data,layout=layout)
iplot(fig, filename='grouped-bar')




trace=[0]*6
purpose=sorted(df['purpose'].unique())
for i in range(0,(df['purpose'].nunique())-1):
    trace[i] = go.Box(x=df['int_rate'][df['purpose']==purpose[i]],name = purpose[i].title())
layout=go.Layout(title='Distribution of Interest Rates by Purpose',xaxis=dict(title='Interest Rates by Purpose'))
fig=go.Figure(data=trace,layout=layout)
iplot(fig)




df2=df.copy()

state_average_int_rate=df.groupby('addr_state').agg({'int_rate':np.average,'id':np.count_nonzero,'annual_inc':np.average})
state_average_int_rate.reset_index(inplace=True)
state_average_int_rate['id']=state_average_int_rate['id'].astype(str)
state_average_int_rate['interest']=state_average_int_rate['int_rate']
state_average_int_rate['int_rate']= 'Average Interest Rate: ' + state_average_int_rate['int_rate'].apply(lambda x: str(round(x,2)))+ "%"
state_average_int_rate['annual_inc']=(state_average_int_rate['annual_inc']/1000.0)
state_average_int_rate['annual_inc']=state_average_int_rate['annual_inc'].apply(lambda x: str(round(x,2)))
state_average_int_rate['text']='Number of Applicants: ' + state_average_int_rate['id']+'<br>'+ 'Average Annual Inc: $'+ state_average_int_rate['annual_inc']+'k'

scl= [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_average_int_rate['addr_state'],
        z = state_average_int_rate['interest'].astype(float),
        text=state_average_int_rate['text'],
        locationmode = 'USA-states',
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Interest Rates")
        ) ]

layout = dict(
        title = '<b>Interest Rate by US States</b><br>Additional Details: <br> Avreage Annual Inc \t Number of Applicants',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
        
             ))
    
fig = dict( data=data, layout=layout )
iplot( fig, filename='d3-cloropleth-map' )

